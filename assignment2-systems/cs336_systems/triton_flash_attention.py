# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

def get_tile_sizes(seq_len, D):
    if D <= 64:
        q_tile, k_tile = 32, 64
    else:
        q_tile, k_tile = 16, 32

    if seq_len <= 128:
        k_tile = min(k_tile, 32)
    elif seq_len >= 4096 and D <= 64:
        q_tile = 64

    return q_tile, k_tile

@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr, DO_ptr, DELTA_ptr,
    stride_ob, stride_oq, stride_od, 
    stride_dob, stride_doq, stride_dod, 
    stride_deltab, stride_deltaq,
    N_QUERYS: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr):

    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    o_batch_ptr = O_ptr + batch_idx * stride_ob 
    o_block_ptr = tl.make_block_ptr(
        base=o_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    do_batch_ptr = DO_ptr + batch_idx * stride_dob
    do_block_ptr = tl.make_block_ptr(
        base=do_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    ) 

    delta_batch_ptr = DELTA_ptr + batch_idx * stride_deltab
    delta_block_ptr = tl.make_block_ptr(
        base=delta_batch_ptr,
        shape=(N_QUERYS, ),
        strides=(stride_deltaq, ),
        offsets=(query_tile_idx * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ), 
        order=(0,)
    )

    o = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    delta = tl.sum(o * do, axis=1)  # sum over ALL D columns, not over Q rows
    tl.store(delta_block_ptr, delta, boundary_check=(0,))

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, grad_output_ptr, DELTA_ptr, dk_ptr, dv_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd, 
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_deltab, stride_deltaq, 
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERYS, N_KEYS, 
    scale,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    K_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    is_causal: tl.constexpr,
):
    key_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    q_batch_ptr = Q_ptr + batch_idx * stride_qb 
    query_block_ptr = tl.make_block_ptr(
        base=q_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    k_batch_ptr = K_ptr + batch_idx * stride_kb
    key_block_ptr = tl.make_block_ptr(
        base=k_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    v_batch_ptr = V_ptr + batch_idx * stride_vb
    value_block_ptr = tl.make_block_ptr(
        base=v_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    l_ptrs = L_ptr + batch_idx * stride_lb
    l_block_ptr = tl.make_block_ptr(
        l_ptrs,
        shape=(N_QUERYS,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    gradoutpu_batch_ptr = grad_output_ptr + batch_idx * stride_dob
    gradoutput_block_ptr = tl.make_block_ptr(
        base=gradoutpu_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    delta_batch_ptr = DELTA_ptr + batch_idx * stride_deltab
    delta_block_ptr = tl.make_block_ptr(
        base=delta_batch_ptr,
        shape=(N_QUERYS, ),
        strides=(stride_deltaq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ), 
        order=(0,)
    )
    dk_batch_ptr = dk_ptr + batch_idx * stride_dkb
    dk_block_ptr = tl.make_block_ptr(
        base=dk_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    dv_batch_ptr = dv_ptr + batch_idx * stride_dvb
    dv_block_ptr = tl.make_block_ptr(
        base=dv_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_idx * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    k_tile = tl.load(key_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_tile = tl.load(value_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dk = tl.zeros((K_TILE_SIZE, D), tl.float32)
    dv = tl.zeros((K_TILE_SIZE, D), tl.float32)

    for i in range(tl.cdiv(N_QUERYS, Q_TILE_SIZE)):
        q_tile = tl.load(query_block_ptr, boundary_check=(0, 1), padding_option="zero")
        original_q_tile_dtype = q_tile.dtype
        q_scaled = (q_tile.to(tl.float32) * scale).to(original_q_tile_dtype) 
        l_tile = tl.load(l_block_ptr, boundary_check=(0, ), padding_option="zero")
        do_tile = tl.load(gradoutput_block_ptr, boundary_check=(0, 1), padding_option="zero")
        delta = tl.load(delta_block_ptr, boundary_check=(0,), padding_option="zero")


        scores = tl.dot(q_scaled, tl.trans(k_tile))
        # 计算当前 Query Tile 和 Key Tile 中，每个 Token 在整个 Sequence 中的真实位置
        q_range = (i * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE))
        k_range = (key_tile_idx * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))
        valid = ((q_range[:, None] < N_QUERYS) & (k_range[None, :] < N_KEYS))
        if is_causal:
            causal_mask = q_range[:, None] < k_range[None, :]
            scores = scores + tl.where(causal_mask, -1e6, 0.0)
        scores = tl.where(valid, scores, -float("inf"))
        l_safe = tl.where(l_tile == -float("inf"), 0.0, l_tile)
        p = tl.exp(scores - l_safe[:, None])
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta[:, None])
        dv = tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile, acc=dv)
        dk = tl.dot(tl.trans(ds.to(q_tile.dtype)), q_tile, acc=dk)
        query_block_ptr = query_block_ptr.advance((Q_TILE_SIZE, 0))
        gradoutput_block_ptr = gradoutput_block_ptr.advance((Q_TILE_SIZE, 0))
        l_block_ptr = l_block_ptr.advance((Q_TILE_SIZE, ))
        delta_block_ptr = delta_block_ptr.advance((Q_TILE_SIZE,))

    dk = dk * scale
    tl.store(dk_block_ptr, dk, boundary_check=(0, 1))
    tl.store(dv_block_ptr, dv, boundary_check=(0, 1)) 

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, grad_output_ptr, DELTA_ptr, dq_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd, 
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_deltab, stride_deltaq, 
    stride_dqb, stride_dqk, stride_dqd,
    N_QUERYS, N_KEYS, 
    scale,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    K_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    q_batch_ptr = Q_ptr + batch_idx * stride_qb 
    query_block_ptr = tl.make_block_ptr(
        base=q_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    k_batch_ptr = K_ptr + batch_idx * stride_kb
    key_block_ptr = tl.make_block_ptr(
        base=k_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    v_batch_ptr = V_ptr + batch_idx * stride_vb
    value_block_ptr = tl.make_block_ptr(
        base=v_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    l_ptrs = L_ptr + batch_idx * stride_lb
    l_block_ptr = tl.make_block_ptr(
        l_ptrs,
        shape=(N_QUERYS,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    gradoutpu_batch_ptr = grad_output_ptr + batch_idx * stride_dob
    gradoutput_block_ptr = tl.make_block_ptr(
        base=gradoutpu_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    delta_batch_ptr = DELTA_ptr + batch_idx * stride_deltab
    delta_block_ptr = tl.make_block_ptr(
        base=delta_batch_ptr,
        shape=(N_QUERYS, ),
        strides=(stride_deltaq, ),
        offsets=(query_tile_idx * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ), 
        order=(0,)
    )
    dq_batch_ptr = dq_ptr + batch_idx * stride_dqb
    dq_block_ptr = tl.make_block_ptr(
        base=dq_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_dqk, stride_dqd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    dq = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    q_tile = tl.load(query_block_ptr, boundary_check=(0, 1), padding_option="zero")
    original_q_tile_dtype = q_tile.dtype
    q_scaled = (q_tile.to(tl.float32) * scale).to(original_q_tile_dtype) 
    l_tile = tl.load(l_block_ptr, boundary_check=(0, ), padding_option="zero")
    do_tile = tl.load(gradoutput_block_ptr, boundary_check=(0, 1), padding_option="zero")
    delta = tl.load(delta_block_ptr, boundary_check=(0,), padding_option="zero")
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):    
        k_tile = tl.load(key_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(value_block_ptr, boundary_check=(0, 1), padding_option="zero")
        scores = tl.dot(q_scaled, tl.trans(k_tile))
        # 计算当前 Query Tile 和 Key Tile 中，每个 Token 在整个 Sequence 中的真实位置
        q_range = (query_tile_idx * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE))
        k_range = (i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))
        valid = ((q_range[:, None] < N_QUERYS) & (k_range[None, :] < N_KEYS))
        if is_causal:
            causal_mask = q_range[:, None] < k_range[None, :]
            scores = scores + tl.where(causal_mask, -1e6, 0.0)
        scores = tl.where(valid, scores, -float("inf"))
        l_safe = tl.where(l_tile == -float("inf"), 0.0, l_tile)
        p = tl.exp(scores - l_safe[:, None])
        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(k_tile.dtype), k_tile, acc=dq)
        key_block_ptr = key_block_ptr.advance((K_TILE_SIZE, 0))
        value_block_ptr = value_block_ptr.advance((K_TILE_SIZE, 0))

    dq = dq * scale
    tl.store(dq_block_ptr, dq, boundary_check=(0, 1))
  

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERYS, N_KEYS, 
    scale,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    K_TILE_SIZE: tl.constexpr, # 一个 Triton Program 一次处理多少行
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    #Q Tile 固定：一个 Program 只负责这一组 Query。 K、V Tile 不断移动：每个 Query 都需要与所有 Key 计算 Attention，才能得到最终输出
    q_batch_ptr = Q_ptr + batch_idx * stride_qb
    query_block_ptr = tl.make_block_ptr(
        base=q_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    k_batch_ptr = K_ptr + batch_idx * stride_kb
    key_block_ptr = tl.make_block_ptr(
        base=k_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    v_batch_ptr = V_ptr + batch_idx * stride_vb
    value_block_ptr = tl.make_block_ptr(
        base=v_batch_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    q_tile = tl.load(query_block_ptr, boundary_check=(0, 1), padding_option="zero")
    original_q_tile_dtype = q_tile.dtype
    q_tile = (q_tile.to(tl.float32) * scale).to(original_q_tile_dtype)

    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    if is_causal:
        end = tl.minimum(N_KEYS, (query_tile_idx + 1) * Q_TILE_SIZE)
    else:
        end = N_KEYS

    for i in range(tl.cdiv(end, K_TILE_SIZE)):
        k_tile = tl.load(key_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(value_block_ptr, boundary_check=(0, 1), padding_option="zero")
        scores = tl.dot(q_tile, tl.trans(k_tile))
        # 计算当前 Query Tile 和 Key Tile 中，每个 Token 在整个 Sequence 中的真实位置
        q_range = (query_tile_idx * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE))
        k_range = (i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))
        valid = ((q_range[:, None] < N_QUERYS) & (k_range[None, :] < N_KEYS))
        if is_causal:
            causal_mask = q_range[:, None] < k_range[None, :]
            scores = scores + tl.where(causal_mask, -1e6, 0.0)
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1)) 
        # 对完全没有有效 Key 的行使用安全最大值，
        # 避免 -inf - (-inf) 产生 NaN
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        p_tilde = tl.exp(scores - m_safe[:, None])

        l = alpha * l + tl.sum(p_tilde, axis=1)
        acc = alpha[:, None] * acc
        acc = tl.dot(p_tilde.to(v_tile.dtype), v_tile, acc=acc)
        m = m_new 

        key_block_ptr = key_block_ptr.advance((K_TILE_SIZE, 0))
        value_block_ptr = value_block_ptr.advance((K_TILE_SIZE, 0))

    l_safe = tl.where(l > 0, l, 1.0)
    output = acc / l_safe[:, None]
    logsumexp = tl.where(l > 0, m + tl.log(l_safe), -float("inf"))

    o_batch_ptr = O_ptr + batch_idx * stride_ob

    output_block_ptr = tl.make_block_ptr(
        base=o_batch_ptr,
        shape=(N_QUERYS, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    tl.store(output_block_ptr, output, boundary_check=(0, 1))
    l_ptrs = L_ptr + batch_idx * stride_lb
    L_block_ptr = tl.make_block_ptr(
        l_ptrs,
        shape=(N_QUERYS,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    tl.store(L_block_ptr, logsumexp, boundary_check=(0,))


class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch, num_querys, D = Q.shape
        num_keys = K.shape[1]

        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Our pointer arithmetic will assume contiguous x" 
        
        ctx.is_causal = is_causal
        ctx.Q_ROWS_TILE_SIZE, ctx.K_ROWS_TILE_SIZE = get_tile_sizes(num_querys, D)

        O = torch.empty_like(Q)
        L = torch.empty((batch, num_querys), device=Q.device, dtype=torch.float32)
        scale = D ** -0.5

        flash_fwd_kernel[(triton.cdiv(num_querys, ctx.Q_ROWS_TILE_SIZE), batch)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1), 
            num_querys, num_keys,
            scale,
            D,
            ctx.Q_ROWS_TILE_SIZE,
            ctx.K_ROWS_TILE_SIZE,
            ctx.is_causal  
        )
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L, Q, K, V, O = ctx.saved_tensors
        batch, num_querys, D = Q.shape
        num_keys = K.shape[1]
        scale = D ** -0.5

        q_tile = ctx.Q_ROWS_TILE_SIZE
        k_tile = ctx.K_ROWS_TILE_SIZE

        grad_output = grad_output.contiguous()
        delta = torch.empty((batch, num_querys), device=Q.device, dtype=torch.float32)
        dq = torch.empty_like(Q)
        dk = torch.empty_like(K)
        dv = torch.empty_like(V)

        flash_bwd_preprocess_kernel[(triton.cdiv(num_querys, q_tile), batch)](
            O, grad_output, delta,
            *O.stride(),
            *grad_output.stride(),
            *delta.stride(),
            num_querys, D, q_tile,
        )
        flash_bwd_dkdv_kernel[(triton.cdiv(num_keys, k_tile), batch)](
            Q, K, V, L, grad_output, delta, dk, dv,
            *Q.stride(), *K.stride(), *V.stride(),
            *L.stride(), *grad_output.stride(),
            *delta.stride(), *dk.stride(), *dv.stride(),
            num_querys, num_keys,
            scale, D, q_tile, k_tile, ctx.is_causal,
        )
        flash_bwd_dq_kernel[(triton.cdiv(num_querys, q_tile), batch)](
            Q, K, V, L, grad_output, delta, dq,
            *Q.stride(), *K.stride(), *V.stride(),
            *L.stride(), *grad_output.stride(),
            *delta.stride(), *dq.stride(),
            num_querys, num_keys,
            scale, D, q_tile, k_tile, ctx.is_causal,
        )
        return dq, dk, dv, None

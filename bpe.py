# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""tokenization
    you may not use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
    • torch.nn.Parameter
    • Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1
    • The torch.optim.Optimizer base class
    preferring the lexicographically greater pair
    """

import os
import torch
import regex as re
from collections import Counter, defaultdict
import multiprocessing
from typing import BinaryIO
import multiprocessing

_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
_SPECIAL_PATTERN_CACHE = None

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def read_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_basic_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")
    return vocab

def cache_special_tokens(special_tokens):
    global _SPECIAL_PATTERN_CACHE
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if _SPECIAL_PATTERN_CACHE is None:
        _SPECIAL_PATTERN_CACHE = re.compile(pattern)

def split_by_special(text, special_tokens):
    """
    Split the text into chunks based on the special tokens.
    Returns a list of chunks, where each chunk is a string.
    """
    if not special_tokens:
        return [text]
    if _SPECIAL_PATTERN_CACHE is None:
        cache_special_tokens(special_tokens)
    chunks = _SPECIAL_PATTERN_CACHE.split(text)
    return [c for c in chunks if c]

def word2bytes(word):
    "convert string to utf-8 bytes"
    byte_list = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in byte_list)

def token_count(text):
    word_cnt = defaultdict(int)
    for m in _PAT.finditer(text):
        word = m.group(0)
        word_bytes = word2bytes(word)
        word_cnt[word_bytes]+=1
    return word_cnt

def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            merged[k] += v
    return merged

def pre_tokenization(text, special_tokens):
    chunks = split_by_special(text, special_tokens)
    word_dicts = list(map(token_count, chunks))
    word_freqs = merge_dicts(word_dicts)
    return word_freqs

def pretokenize_chunk(args):
    chunk_text, special_tokens = args
    return pre_tokenization(chunk_text, special_tokens)

def build_index(word_freqs):
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)   # pair -> set of word_id
    for word_bytes, freq in word_freqs.items():
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_bytes)
    return pair_counts, pair_to_words

def _remove_word_pairs(word_bytes, freq, pair_counts, pair_to_words):
    for i in range(len(word_bytes) - 1):
        pair = (word_bytes[i], word_bytes[i + 1])
        pair_counts[pair] -= freq
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
        s = pair_to_words.get(pair)
        if s is not None:
            s.discard(word_bytes)
            if not s:
                del pair_to_words[pair]

def _add_word_pairs(word_bytes, freq, pair_counts, pair_to_words):
    for i in range(len(word_bytes) - 1):
        pair = (word_bytes[i], word_bytes[i + 1])
        pair_counts[pair] += freq
        pair_to_words[pair].add(word_bytes)

def _merge_word(word_bytes, max_pair):
    i = 0
    new_word = []
    while i < len(word_bytes):
        if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i + 1]) == max_pair:
            new_word.append(word_bytes[i] + word_bytes[i + 1])
            i += 2
        else:
            new_word.append(word_bytes[i])
            i += 1
    return tuple(new_word)

def get_max(pair_count):
    max_pair, count = max(pair_count.items(), key=lambda x: (x[1], x[0])) 
    return max_pair

def train_bpe(input_path, vocab_size, special_tokens):
    # text = read_text(input_path)
    num_processes = 8
    word_freqs = defaultdict(int)
    chunk_texts = []
    cache_special_tokens(special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*4, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            chunk_texts.append(chunk)
        
    with multiprocessing.Pool(num_processes) as pool:
        word_dicts = pool.map(pretokenize_chunk, [(ct, special_tokens) for ct in chunk_texts])
    
    word_freqs = merge_dicts(word_dicts)

    # 构建索引
    pair_counts, pair_to_words = build_index(word_freqs)

    vocab = get_basic_vocab(special_tokens)
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size

    merges = []
    for i in range(n_merges):
        if not pair_counts:
            break
        max_pair = get_max(pair_counts)
        new_token = max_pair[0] + max_pair[1]
        vocab[base_vocab_size + i] = new_token
        merges.append(max_pair)

        affected_words = list(pair_to_words.pop(max_pair, ()))
        if not affected_words:
            continue

        new_word_freqs = defaultdict(int)

        for word_bytes in affected_words:
            freq = word_freqs.pop(word_bytes, 0)
            if freq == 0:
                continue
            _remove_word_pairs(word_bytes, freq, pair_counts, pair_to_words)
            new_word = _merge_word(word_bytes, max_pair)
            _add_word_pairs(new_word, freq, pair_counts, pair_to_words)

            new_word_freqs[new_word] += freq

        for w, f in new_word_freqs.items():
            word_freqs[w] = word_freqs.get(w, 0) + f

    special_token_set = set(tok.encode("utf-8") for tok in special_tokens)
    longest_word = max(
        (v for v in vocab.values() if v not in special_token_set),
        key=len
    )
    print(f"Longest word in vocab: {longest_word} (length {len(longest_word)})")

    return vocab, merges


if __name__ == "__main__":
    train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])

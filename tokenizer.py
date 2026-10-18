# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import torch
import functools
import pickle
import heapq
import regex as re
from typing import BinaryIO
import multiprocessing
import numpy as np
import time
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Type

class Tokenizer:
    """
    BPE-style tokenizer.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge2rank: Dict[Tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(merges)}

        self.token_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.special_pattern_str = "(?!)" 
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern_str = "(" + "|".join(re.escape(tok) for tok in sorted_special) + ")"
        self.special_tokens_pattern = re.compile(self.special_pattern_str)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: Optional[List[str]] = None) -> "Tokenizer":
        """
        Load a tokenizer from vocab and merges files.
        """
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        print(f"vocab size: {len(vocab)}")
        print(f"merges size: {len(merges)}")
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def word2bytes(self, word):
        # tuple of 1-byte bytes objects; consistent with your original representation
        return tuple(bytes([b]) for b in word.encode("utf-8"))
    
    def token_merges(self, word_bytes):
        if len(word_bytes) < 2:
            return word_bytes
        
        word = list(word_bytes)
        n = len(word)
        prev = [i - 1 for i in range(n)]
        next_ = [i + 1 for i in range(n)]
        prev[0] = -1
        next_[-1] = -1

        heap = []
        def push_pair(left):
            """Push (left, right) into heap if it is currently mergeable."""
            if left == -1:
                return

            right = next_[left]
            if right == -1:
                return

            pair = (word[left], word[right])
            rank = self.merge2rank.get(pair)

            if rank is not None:
                heapq.heappush(heap, (rank, left, right))

        for i in range(n - 1):
            push_pair(i) 
        
        while heap:
            rank, left, right = heapq.heappop(heap)
            if next_[left] != right:
                continue
            if prev[right] != left:
                continue

            pair = (word[left], word[right])
            curr_rank = self.merge2rank.get(pair)
            if curr_rank != rank:
                continue

            word[left] = word[left] + word[right]
            right_next = next_[right]
            next_[left] = right_next
            if right_next != -1:
                prev[right_next] = left
            # `right` is now dead.
            prev[right] = -2
            next_[right] = -2
            # 更新左边、右边
            left_prev = prev[left]
            push_pair(left_prev)
            push_pair(left)

        result = []
        node = 0
        while node != -1:
            result.append(word[node])
            node = next_[node]

        return tuple(result)

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        token_ids = []
        for segment in self.special_tokens_pattern.split(text):
            if not segment:
                continue
            if segment in self.special_tokens:
                token_ids.append(self.token2id[segment.encode("utf-8")])
                continue
            for match in self.token_pattern.finditer(segment):
                word = match.group(0)
                merged_token = self.token_merges(self.word2bytes(word))
                tmp_token_ids = [self.token2id[token] for token in merged_token]
                token_ids.extend(tmp_token_ids)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a file handle yielding lines), return a
        generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large inputs that cannot
        be fully loaded into memory.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = [self.vocab[token_id] for token_id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")



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


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "../data/bpe_vocab.pkl",
        "../data/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    # input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 8
    special_tokens = ["<|endoftext|>"]

    chunk_texts = []
    bytes_len = 0
    start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*4, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk = chunk_bytes.decode("utf-8")
            bytes_len += len(chunk_bytes)
            chunk_texts.append(chunk)
    
    with multiprocessing.Pool(num_processes) as pool:
        token_ids_pool = pool.map(tokenizer.encode, chunk_texts)
    all_token_ids = [tid for chunk in token_ids_pool for tid in chunk]
    elapsed = time.time() - start_time

    token_ids_np = np.array(all_token_ids, dtype=np.uint16)
    # np.save("../data/token_ids_tiny_stories.npy", token_ids_np)
    np.save("../data/token_ids_tiny_stories_valid.npy", token_ids_np)

    
    num_tokens = len(all_token_ids)
    print(f"Compression ratio: {bytes_len/num_tokens:.2f}, byte_len: {bytes_len}")
    throughput = bytes_len * 1.0 / elapsed   # bytes/second
    print(f"Throughput: {throughput / 1e6:.6f} MB/s")
    print(f"Tokens: {num_tokens}")
    print(f"Tokens/s: {num_tokens / elapsed:.2f}")

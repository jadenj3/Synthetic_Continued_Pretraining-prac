import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
from tqdm import tqdm


def get_tokenizer(tokenizer_model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length = 2**20
    return tokenizer


def tokenize_list(text_list: List[str]) -> List[int]:
    """Tokenize the text and return the tokenized text."""
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    all_ids = []
    for text in tqdm(text_list, desc="Tokenizing"):
        if text:
            ids = tokenizer.encode(text)
            ids.append(tokenizer.eos_token_id)
            all_ids.extend(ids)
    return all_ids


def write_to_memmap_single(ids: List[int], filename: str):
    filename = f'data/dataset/bins/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()


def get_quality_source_articles() -> List[str]:
    """Read the original QuALITY articles from the JSONL file."""
    filepath = 'data/dataset/raw/QuALITY.v1.0.1.htmlstripped.train'
    articles = []
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Reading articles"):
            data = json.loads(line)
            articles.append(data['article'])
    print(f"Loaded {len(articles)} articles")
    return articles


def tokenize_quality_source():
    """Tokenize the original QuALITY source articles."""
    articles = get_quality_source_articles()
    write_to_memmap_single(tokenize_list(articles), 'quality_source.bin')


if __name__ == '__main__':
    tokenize_quality_source()

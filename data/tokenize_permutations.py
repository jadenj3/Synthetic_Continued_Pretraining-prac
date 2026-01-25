import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload


def get_tokenizer(tokenizer_model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length = 2**20
    return tokenizer


def tokenize_list(text_list: List[str]) -> List[int]:
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    all_ids = []
    for text in tqdm(text_list):
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


def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f'{dir_name}/*.json') + glob.glob(f'{dir_name}/.*.json')


def get_permutations(input_folder: str) -> List[str]:
    files = _glob_all_json(input_folder)
    result = []
    for file in files:
        content = jload(file)
        result.extend(content)
    return result


def tokenize_permutations(input_folder: str, output_name: str):
    texts = get_permutations(input_folder)
    print(f"Found {len(texts)} permutations")
    write_to_memmap_single(tokenize_list(texts), output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize DAG permutations')
    parser.add_argument('--input_folder', required=True, help='Folder with permutation JSON files')
    parser.add_argument('--output_name', default='permutations.bin', help='Output bin filename')
    args = parser.parse_args()

    tokenize_permutations(args.input_folder, args.output_name)

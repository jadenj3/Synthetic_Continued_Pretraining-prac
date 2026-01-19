import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import List
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload

def get_tokenizer(tokenizer_model_name: str)-> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning
    return tokenizer

def tokenize_list(text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
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

def _get_quality_graph(model_name: str) -> List[str]:
    files = _glob_all_json(f'data/dataset/raw/quality_entigraph_{model_name}')
    result = []
    for file in files:
        content = jload(file)
        result.extend(content[1:])
    return result

def tokenize_quality_graph(model_name: str):
    quality = _get_quality_graph(model_name)
    write_to_memmap_single(tokenize_list(quality), f'quality_all-entigraph{model_name}.bin')

def _get_quality_graph_from_hf() -> List[str]:
    """Download and extract entigraph data from HuggingFace dataset."""
    ds = load_dataset("zitongyang/entigraph-quality-corpus", split="train")
    result = []
    for row in tqdm(ds, desc="Extracting entigraph chunks"):
        chunks = row["entigraph"].split("<|entigraphseptoekn|>")
        result.extend(chunks)
    return result

def tokenize_quality_graph_hf():
    """Download from HuggingFace and tokenize the entigraph dataset."""
    quality = _get_quality_graph_from_hf()
    write_to_memmap_single(tokenize_list(quality), 'quality_all-entigraphgpt-4-turbo.bin')

if __name__ == '__main__':
    # Usage: python data/tokenize_entigraph.py [--hf]
    # --hf: Download from HuggingFace instead of using local JSON files
    if len(sys.argv) > 1 and sys.argv[1] == '--hf':
        tokenize_quality_graph_hf()
    else:
        tokenize_quality_graph('gpt-4-turbo')
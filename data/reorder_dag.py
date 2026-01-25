import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import json
import glob
import random
import re
from collections import defaultdict, deque
from typing import List, Dict, Any
from utils.io_utils import jdump


def parse_dag_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON DAG from text between <start_dag> and <end_dag> tags."""
    match = re.search(r'<start_dag>\s*(.*?)\s*<end_dag>', text, re.DOTALL)
    if not match:
        raise ValueError("Could not find DAG between <start_dag> and <end_dag> tags")
    return json.loads(match.group(1))


def get_valid_permutations(dag: Dict[str, Any], num_permutations: int) -> List[List[str]]:
    """Generate valid topological orderings of the DAG."""
    nodes = {n['id']: n['content'] for n in dag['nodes']}
    node_ids = list(nodes.keys())

    # Build adjacency list and in-degree count
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    for node_id in node_ids:
        in_degree[node_id] = 0

    for edge in dag['edges']:
        adj[edge['source']].append(edge['target'])
        in_degree[edge['target']] += 1

    permutations = []
    seen = set()

    for _ in range(num_permutations * 10):  # Try more times to get unique permutations
        if len(permutations) >= num_permutations:
            break

        # Kahn's algorithm with random selection among available nodes
        result = []
        temp_in_degree = dict(in_degree)
        available = [n for n in node_ids if temp_in_degree[n] == 0]

        while available:
            random.shuffle(available)
            node = available.pop()
            result.append(node)

            for neighbor in adj[node]:
                temp_in_degree[neighbor] -= 1
                if temp_in_degree[neighbor] == 0:
                    available.append(neighbor)

        # Check if valid (all nodes included)
        if len(result) == len(node_ids):
            result_tuple = tuple(result)
            if result_tuple not in seen:
                seen.add(result_tuple)
                permutations.append(result)

    return permutations


def permutation_to_text(dag: Dict[str, Any], permutation: List[str]) -> str:
    """Convert a permutation of node IDs to concatenated sentences."""
    nodes = {n['id']: n['content'] for n in dag['nodes']}
    sentences = [nodes[node_id] for node_id in permutation]
    return ' '.join(sentences)


def process_dag_file(input_path: str, num_permutations: int) -> List[str]:
    """Process a single DAG file and return permuted texts."""
    with open(input_path, 'r') as f:
        content = f.read()

    # Handle JSON list format from gemini_generate.py
    try:
        data = json.loads(content)
        if isinstance(data, list):
            content = data[0] if data else ""
    except json.JSONDecodeError:
        pass  # Not JSON, use raw content

    dag = parse_dag_from_text(content)
    permutations = get_valid_permutations(dag, num_permutations)

    texts = [permutation_to_text(dag, perm) for perm in permutations]
    return texts


def reorder_documents(
    input_folder: str,
    output_folder: str,
    num_permutations: int = 5
):
    """Process all DAG files and generate permutations."""
    input_files = [f for f in glob.glob(os.path.join(input_folder, '*'))
                   if os.path.isfile(f)]

    os.makedirs(output_folder, exist_ok=True)

    for input_path in input_files:
        basename = os.path.basename(input_path)
        try:
            texts = process_dag_file(input_path, num_permutations)
            output_path = os.path.join(output_folder, f"{basename}.json")
            jdump(texts, output_path)
            print(f"Processed: {basename} -> {len(texts)} permutations")
        except Exception as e:
            print(f"Error processing {basename}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate permutations from DAG documents')
    parser.add_argument('--input_folder', required=True, help='Folder with DAG JSON files')
    parser.add_argument('--output_folder', required=True, help='Folder for output permutations')
    parser.add_argument('--num_permutations', type=int, default=5, help='Number of permutations per document')
    args = parser.parse_args()

    reorder_documents(args.input_folder, args.output_folder, args.num_permutations)

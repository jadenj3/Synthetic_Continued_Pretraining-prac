import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re
from tasks.quality import QuALITY


def sanitize_filename(title: str) -> str:
    """Remove/replace characters invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', title)


def extract_articles(split: str, output_folder: str):
    task = QuALITY(split)
    os.makedirs(output_folder, exist_ok=True)

    for doc in task.documents:
        filename = sanitize_filename(doc.title) + '.txt'
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w') as f:
            f.write(doc.text)
        print(f"Extracted: {filename}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract QuALITY articles to text files')
    parser.add_argument('--split', default='train', help='Dataset split (train/dev/all)')
    parser.add_argument('--output_folder', default='data/dataset/raw/documents', help='Output folder')
    args = parser.parse_args()

    extract_articles(args.split, args.output_folder)

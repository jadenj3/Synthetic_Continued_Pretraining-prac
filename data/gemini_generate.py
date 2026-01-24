import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import glob
import asyncio
from datetime import datetime
from tqdm.asyncio import tqdm
from google import genai
from utils.io_utils import jdump


async def generate_single(
    client: genai.Client,
    prompt: str,
    input_path: str,
    output_folder: str,
    model: str
):
    """Generate synthetic data for a single document."""
    # Read document content
    with open(input_path, 'r') as f:
        content = f.read()

    # Combine prompt + content
    full_prompt = f"{prompt}\n\n{content}"

    # Call Gemini API (async)
    response = await client.aio.models.generate_content(
        model=model,
        contents=full_prompt,
    )

    # Output as JSON list (compatible with tokenization pipeline)
    basename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, f"{basename}.json")
    output = [response.text]
    jdump(output, output_path)

    return output_path


async def generate_with_gemini(
    prompt_file: str,
    input_folder: str,
    output_folder: str,
    model: str = "gemini-2.5-flash-lite",
    batch_size: int = 10
):
    # Add timestamp suffix to output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{output_folder}_{timestamp}"

    # Read prompt template
    with open(prompt_file, 'r') as f:
        prompt = f.read()

    # Get all files in input folder
    input_files = [f for f in glob.glob(os.path.join(input_folder, '*'))
                   if os.path.isfile(f)]

    # Initialize Gemini client
    client = genai.Client()

    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Process in batches
    for i in range(0, len(input_files), batch_size):
        batch = input_files[i:i + batch_size]
        tasks = [
            generate_single(client, prompt, input_path, output_folder, model)
            for input_path in batch
        ]
        results = await tqdm.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        for path in results:
            print(f"Generated: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic data using Gemini API')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--input_folder', required=True, help='Folder with input documents')
    parser.add_argument('--output_folder', required=True, help='Base folder for output (timestamp added)')
    parser.add_argument('--model', default='gemini-2.5-flash-lite', help='Gemini model name')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of concurrent API calls')
    args = parser.parse_args()

    asyncio.run(generate_with_gemini(
        args.prompt, args.input_folder, args.output_folder, args.model, args.batch_size
    ))

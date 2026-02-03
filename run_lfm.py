import json
import uuid
from pathlib import Path

from vllm import LLM, SamplingParams

DATA_DIR = Path("lfm_data")
OUTPUT_DIR = Path("data/dataset/lfm_outputs")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the prompt template
    with open("lfm_prompt.txt", "r") as f:
        prompt_template = f.read().strip()

    if not prompt_template:
        print("No prompt found in lfm_prompt.txt")
        return

    # Read all data files from lfm_data/
    data_files = sorted(DATA_DIR.glob("*.txt"))
    if not data_files:
        print(f"No .txt files found in {DATA_DIR}/")
        return

    # Build prompts by combining template with each data file
    prompts = []
    data_sources = []
    for data_file in data_files:
        data_content = data_file.read_text().strip()
        full_prompt = prompt_template + "\n\n" + data_content
        prompts.append(full_prompt)
        data_sources.append(data_file.name)

    print(f"Processing {len(prompts)} prompts...")

    sampling_params = SamplingParams(
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_tokens=1024,
    )

    llm = LLM(model="LiquidAI/LFM2-2.6B-Exp")

    outputs = llm.generate(prompts, sampling_params)

    for output, source_file in zip(outputs, data_sources):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        # Generate unique ID and save to JSON
        output_id = str(uuid.uuid4())
        output_data = {
            "id": output_id,
            "source_file": source_file,
            "prompt": prompt,
            "text": generated_text,
        }

        output_path = OUTPUT_DIR / f"{output_id}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Source: {source_file}")
        print(f"Generated text: {generated_text[:100]!r}...")
        print(f"Saved to: {output_path}")
        print()


if __name__ == "__main__":
    main()

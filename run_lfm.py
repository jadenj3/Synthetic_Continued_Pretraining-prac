import json
import uuid
from pathlib import Path

from vllm import LLM, SamplingParams

OUTPUT_DIR = Path("data/dataset/lfm_outputs")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read prompts from file (one per line)
    with open("lfm_prompt.txt", "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("No prompts found in lfm_prompt.txt")
        return

    sampling_params = SamplingParams(
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05
    )

    llm = LLM(model="LiquidAI/LFM2-2.6B-Exp")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        # Generate unique ID and save to JSON
        output_id = str(uuid.uuid4())
        output_data = {
            "id": output_id,
            "prompt": prompt,
            "text": generated_text,
        }

        output_path = OUTPUT_DIR / f"{output_id}.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Saved to: {output_path}")
        print()


if __name__ == "__main__":
    main()

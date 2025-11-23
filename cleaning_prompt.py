import json
import jsonlines
import argparse
from pathlib import Path


def build_cleaning_prompt(example):
    """
    Build the MOE training prompt for CLEANING tasks.
    """

    purpose = example["purpose"]
    raw_table = example["raw_table"]

    prompt = (
        "[CLEANING]\n"
        "You are a data-cleaning assistant.\n"
        f"Purpose: {purpose}\n\n"
        "Raw Table:\n"
        f"{raw_table}\n\n"
        "Generate:\n"
        "- cleaning_workflow (a JSON list of transformation steps)\n"
        "- clean_table (CSV after cleaning)\n\n"
        "Respond ONLY with a JSON object containing both fields."
    )

    return prompt


def build_cleaning_output(example):
    """
    Build the training target string.
    This is exactly what the model must learn to output.
    """

    output = {
        "cleaning_workflow": example["cleaning_workflow"],
        "clean_table": example["clean_table"]
    }

    # Always dump JSON compactly to avoid formatting drift
    return json.dumps(output, ensure_ascii=False)


def convert_dataset(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = []

    # read line-by-line JSON objects
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    with jsonlines.open(output_path, "a") as writer:
        for ex in data:
            prompt = build_cleaning_prompt(ex)
            output = build_cleaning_output(ex)
            writer.write({
                "prompt": prompt,
                "output": output
            })

    print(f"Finished. Wrote {len(data)} examples to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to your dataset.json")
    parser.add_argument("--output", required=True, help="output jsonl path")
    args = parser.parse_args()

    convert_dataset(args.input, args.output)

import argparse
import json
import os
import re
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

RESPONSES_URL = "https://openrouter.ai/api/v1/responses"

# run like
# uv run classify_create_codebook.py --model "anthropic/claude-sonnet-5" --nametag "elemental" --note "helpful note" --data-path "==PATH==" --prompt-path "==PATH==" --test-rows ==e.g. 5, if you want to test 5 rows==
# will output a .csv with the classifications, and a .json with the codebook

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nametag", default="", help="add a descriptive tag to add to output names, like 'elemental'")
    parser.add_argument("--note", default=None, help="add a note if you want")
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev_350.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/prompt_classify-create-codebook_grimmer_elemental.txt"))
    parser.add_argument("--test-rows", type=int, default=None, help="if you want to test a few random rows")
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    return parser.parse_args()


def safe_name(value: str) -> str:
    # make name safe for a file name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_response_text(body: dict) -> str:
    message = next(item for item in body["output"] if item["type"] == "message")
    return message["content"][0]["text"]


def clean_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1]


def send_request(api_key: str, request_body: dict) -> dict:
    response = requests.post(
        RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=600,
    )
    body = response.json()
    return body


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    prompt = args.prompt_path.read_text(encoding="utf-8")
    data = pl.read_csv(args.data_path)

    # if you are sampling rows for a test run
    if args.test_rows is not None:
        data = data.sample(n=args.test_rows, seed=1)

    # format data into a array of json objects, each one is a text
    documents = [
        {"document_id": str(row[args.id_column]), "text": str(row[args.text_column])}
        for row in data.iter_rows(named=True)
    ]

    print(
        f"Classifying {len(documents)} documents and generating a codebook "
        f"with {args.model}..."
    )

    response_body = send_request(
        api_key,
        {
            "model": args.model,
            "instructions": prompt,
            "input": json.dumps(documents),
            "temperature": 0,
            "max_output_tokens": args.max_output_tokens,
        },
    )

    # prep file names
    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag)
    rows_folder = (
        "all-rows"
        if args.test_rows is None
        else f"test-{args.test_rows}-rows"
    )
    output_folder = (
        args.out_folder
        / f"tag-{nametag}"
        / f"model-{model_name}"
        / rows_folder
    )
    file_prefix = f"{model_name}_{nametag}"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("saving raw output in", output_folder)
    output_text = get_response_text(response_body)
    # save raw output first before parsing anything, in case there's an issue
    raw_output_path = output_folder / f"{file_prefix}_raw_output.txt"
    raw_output_path.write_text(output_text, encoding="utf-8")

    classifications_path = output_folder / f"{file_prefix}_classifications.csv"
    codebook_path = output_folder / f"{file_prefix}_codebook.json"

    result = json.loads(clean_json(output_text))
    classifications = pl.DataFrame(result["classifications"]).with_columns(
        pl.col("document_id").cast(pl.String)
    )
    original_texts = pl.DataFrame(documents)
    classifications.join(original_texts, on="document_id", how="left").select(
        "document_id", "text", "label"
    ).write_csv(classifications_path)
    codebook_path.write_text(
        json.dumps(result["codebook"], indent=2) + "\n",
        encoding="utf-8",
    )

    if args.note is not None:
        (output_folder / f"{file_prefix}_note.txt").write_text(args.note)

    print("parsed and saved classifications and codebook")

if __name__ == "__main__":
    main()

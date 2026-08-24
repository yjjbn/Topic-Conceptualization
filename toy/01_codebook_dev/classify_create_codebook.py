import argparse
import json
import os
import re
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

RESPONSES_URL = "https://openrouter.ai/api/v1/responses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the codebook-development documents and create a codebook."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--nametag",
        default="",
        help="Optional tag added to output filenames, such as 'experiment-1'.",
    )
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev_350.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/prompt_classify-create-codebook_grimmer_elemental.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--test-rows", type=int, default=None)
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_response_text(body: dict) -> str:
    message = next(item for item in body["output"] if item["type"] == "message")
    return message["content"][0]["text"]


def clean_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"error in response format:\n{text}")
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

    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text}")

    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"OpenRouter response error: {body['error']}")
    if body.get("status") == "incomplete":
        raise RuntimeError(
            f"Response was incomplete: {body.get('incomplete_details')}"
        )
    return body


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    prompt = args.prompt_path.read_text(encoding="utf-8")
    data = pl.read_csv(args.data_path)

    if args.test_rows is not None:
        data = data.sample(n=args.test_rows, seed=1)

    documents = [
        {"id": str(row[args.id_column]), "text": str(row[args.text_column])}
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
            "input": json.dumps(documents, ensure_ascii=False),
            "temperature": 0,
            "max_output_tokens": args.max_output_tokens,
        },
    )

    output_text = get_response_text(response_body)
    result = json.loads(clean_json(output_text))

    args.out_folder.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag).strip("._-")
    if nametag:
        model_name = f"{model_name}_{nametag}"
    if args.test_rows is not None:
        model_name = f"{model_name}_{args.test_rows}-rows"
    classifications_path = (
        args.out_folder / f"{model_name}_classifications.csv"
    )
    codebook_path = args.out_folder / f"{model_name}_codebook.json"

    classifications = pl.DataFrame(result["classifications"]).with_columns(
        pl.col("id").cast(pl.String)
    )
    original_texts = pl.DataFrame(documents)
    classifications.join(original_texts, on="id", how="left").select(
        "id", "text", "label"
    ).write_csv(classifications_path)
    codebook_path.write_text(
        json.dumps(result["codebook"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Classifications:", classifications_path)
    print("Codebook:", codebook_path)


if __name__ == "__main__":
    main()

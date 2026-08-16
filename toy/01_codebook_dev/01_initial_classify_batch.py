import argparse
import json
import os
import re
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev.csv"))
    parser.add_argument("--labels", nargs="+", default=["1", "0"])
    parser.add_argument("--classification-prompt-path", type=Path, default=Path("./prompts/initial_classify_prompt_grimmer.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def request_json(method: str, url: str, api_key: str, body: dict, max_retries: int) -> dict:
    for attempt in range(max_retries + 1):
        response = requests.request(
            method,
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(body, ensure_ascii=False),
            timeout=600,
        )

        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_retries:
            break

        try:
            delay = float(response.headers.get("Retry-After"))
        except (TypeError, ValueError):
            delay = 30

        print(f"Retrying in {delay} seconds")
        time.sleep(delay)

    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text}")

    return response.json()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    documents = pl.read_csv(args.data_path)

    classification_prompt = args.classification_prompt_path.read_text(encoding="utf-8")
    args.out_folder.mkdir(parents=True, exist_ok=True)

    model_name = safe_name(args.model)
    classifications_path = args.out_folder / f"initial_classifications_{model_name}.json"
    
    requests_list = []
    submitted_documents = []
    all_document_ids = []

    for position, document in enumerate(documents.iter_rows(named=True), start=1):
        document_id = str(document[args.id_column])
        document_text = str(document[args.text_column])
        all_document_ids.append(document_id)

        custom_id = f"doc-{position:08d}"

        requests_list.append({
            "custom_id": custom_id,
            "body": {
                "messages": [
                    {"role": "system", "content": classification_prompt},
                    {"role": "user", "content": f"Document:\n\n{document_text}"},
                ],
                "temperature": 0,
                "max_tokens": args.max_output_tokens,
            },
        })

        submitted_documents.append({
            "custom_id": custom_id,
            "document_id": document_id,
            "text": document_text,
        })

    print(f"Submitting {len(requests_list)} classification requests...")

    # Keep endpoint and model before requests in this dictionary.
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "requests": requests_list,
    }

    submission = request_json("POST", BATCHES_URL, api_key, payload, args.max_retries)

    if submission.get("error"):
        raise RuntimeError(f"Batch submission error: {submission['error']}")

    batch_id = submission.get("id")
    if not batch_id:
        raise RuntimeError(f"Batch submission did not return an ID: {submission}")

    meta_path = args.out_folder / f"{model_name}_{batch_id}_meta.json"

    meta = {
        "batch_id": batch_id,
        "model": args.model,
        "labels": args.labels,
        "id_column": args.id_column,
        "text_column": args.text_column,
        "data_path": str(args.data_path.resolve()),
        "classifications_path": str(classifications_path.resolve()),
        "all_document_ids": all_document_ids,
        "submitted_documents": submitted_documents,
        "submission": submission,
    }

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Batch ID:", batch_id)
    print("Status:", submission.get("status"))
    print("Manifest:", meta_path)
    print(f"\nRetrieve later with:\nuv run python 01_retrieve_batches.py --meta-path \"{meta_path}\"")


if __name__ == "__main__":
    main()
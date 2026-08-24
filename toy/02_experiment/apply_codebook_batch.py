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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--nametag",
        default="",
        help="Optional tag added to output filenames, such as 'experiment-1'.",
    )
    parser.add_argument("--codebook-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookapply.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/apply_codebook_grimmer_elemental.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def request_json(method, url, api_key, body, max_retries):
    for attempt in range(max_retries + 1):
        response = requests.request(
            method, url,
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


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    documents = pl.read_csv(args.data_path)
    codebook = json.loads(args.codebook_path.read_text(encoding="utf-8"))
    application_prompt = args.prompt_path.read_text(encoding="utf-8").strip()

    instructions = (
        f"{application_prompt}\n\n"
        "The codebook is below, formatted as a JSON object:\n\n"
        f"{json.dumps(codebook, ensure_ascii=False, indent=2)}"
    )

    requests_list, submitted_documents, all_document_ids = [], [], []

    for position, document in enumerate(documents.iter_rows(named=True), start=1):
        document_id = str(document[args.id_column])
        document_text = str(document[args.text_column])
        custom_id = f"doc-{position:08d}"
        all_document_ids.append(document_id)

        model_input = {"id": document_id, "text": document_text}

        requests_list.append({
            "custom_id": custom_id,
            "body": {
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(model_input, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_completion_tokens": args.max_output_tokens,
                "reasoning": {"effort": "medium", "exclude": True},
                "response_format": {"type": "json_object"},
            },
        })

        submitted_documents.append({
            "custom_id": custom_id,
            "document_id": document_id,
            "text": document_text,
        })

    print(f"Submitting {len(requests_list)} codebook application requests...")

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

    args.out_folder.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag).strip("._-")
    if nametag:
        model_name = f"{model_name}_{nametag}"
    codebook_name = safe_name(args.codebook_path.stem)
    output_name = f"model_{model_name}_codebook_{codebook_name}_results"
    csv_path = args.out_folder / f"{output_name}.csv"
    meta_path = args.out_folder / f"{output_name}_{batch_id}_meta.json"

    meta = {
        "batch_id": batch_id,
        "model": args.model,
        "nametag": args.nametag,
        "id_column": args.id_column,
        "text_column": args.text_column,
        "data_path": str(args.data_path.resolve()),
        "codebook_path": str(args.codebook_path.resolve()),
        "output_path": str(csv_path.resolve()),
        "all_document_ids": all_document_ids,
        "submitted_documents": submitted_documents,
        "submission": submission,
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Batch ID:", batch_id)
    print("Status:", submission.get("status"))
    print("Manifest:", meta_path)
    print(f'\nRetrieve later with:\nuv run python retrieve_codebook_batch.py --meta-path "{meta_path}"')


if __name__ == "__main__":
    main()

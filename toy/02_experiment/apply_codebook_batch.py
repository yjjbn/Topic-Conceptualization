import argparse
import json
import os
import re
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
DOCUMENTS_PER_REQUEST = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nametag", default="none", help="tag used to organize output folders, such as 'elemental'")
    parser.add_argument("--note", default=None, help="add a note if you want")
    parser.add_argument("--codebook-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookapply.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/apply_codebook_grimmer_elemental.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--test-rows", type=int, default=None)
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def request_json(method, url, api_key, body):
    response = requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"},
        data=json.dumps(body, ensure_ascii=False),
        timeout=600,
    )
    return response.json()


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    documents = pl.read_csv(args.data_path)
    if args.test_rows is not None:
        documents = documents.sample(n=args.test_rows, seed=1)

    codebook = json.loads(args.codebook_path.read_text(encoding="utf-8"))
    application_prompt = args.prompt_path.read_text(encoding="utf-8")

    instructions = (
        f"{application_prompt}\n\n"
        "The codebook is below, formatted as a JSON object:\n\n"
        f"{json.dumps(codebook, indent=2)}\n\n"
        "Apply the codebook independently to every document and return a JSON "
        "array containing one result object per input document."
    )

    document_inputs = [
        {
            "document_id": str(document[args.id_column]),
            "text": str(document[args.text_column]),
        }
        for document in documents.iter_rows(named=True)
    ]
    requests_list, submitted_requests = [], []

    for start in range(0, len(document_inputs), DOCUMENTS_PER_REQUEST):
        request_documents = document_inputs[start : start + DOCUMENTS_PER_REQUEST]
        request_number = start // DOCUMENTS_PER_REQUEST + 1
        custom_id = f"request-{request_number:08d}"

        requests_list.append({
            "custom_id": custom_id,
            "body": {
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(request_documents, ensure_ascii=False),
                    },
                ],
                "temperature": 0,
                "max_completion_tokens": args.max_output_tokens
            },
        })

        submitted_requests.append({
            "custom_id": custom_id,
            "documents": request_documents,
        })

    print(
        f"Submitting {len(requests_list)} requests for "
        f"{len(document_inputs)} documents..."
    )

    payload = {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "requests": requests_list,
    }

    submission = request_json("POST", BATCHES_URL, api_key, payload, args.max_retries)
    batch_id = submission.get("id")

    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag)
    codebook_name = safe_name(args.codebook_path.stem)
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
        / f"codebook-{codebook_name}"
        / f"batchID-{batch_id}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{model_name}_{nametag}"

    if args.note is not None:
        (output_folder / f"{file_prefix}_note.txt").write_text(args.note)

    results_path = output_folder / f"{file_prefix}_results.csv"
    meta_path = output_folder / f"{file_prefix}_batch_meta.json"

    meta = {
        "batch_id": batch_id,
        "model": args.model,
        "nametag": args.nametag,
        "test_rows": args.test_rows,
        "note": args.note,
        "id_column": args.id_column,
        "text_column": args.text_column,
        "data_path": str(args.data_path.resolve()),
        "codebook_path": str(args.codebook_path.resolve()),
        "output_folder": str(output_folder.resolve()),
        "output_path": str(results_path.resolve()),
        "documents_per_request": DOCUMENTS_PER_REQUEST,
        "all_document_ids": [document["document_id"] for document in document_inputs],
        "submitted_requests": submitted_requests,
        "submission": submission,
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Batch ID:", batch_id)
    print("Status:", submission.get("status"))
    print("Meta:", meta_path)
    print(f'\nRetrieve later with:\nuv run python retrieve_codebook_batch.py --meta-path "{meta_path}"')


if __name__ == "__main__":
    main()

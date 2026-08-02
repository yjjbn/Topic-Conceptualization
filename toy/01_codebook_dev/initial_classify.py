import argparse
import json
import os
import re
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/populism_codebookdev.csv"))
    parser.add_argument("--classifications-path", type=Path)
    parser.add_argument("--classification-prompt-path", type=Path, default=Path("./prompts/initial_classify_prompt.txt"))
    parser.add_argument("--codebook-prompt-path", type=Path, default=Path("./prompts/write_codebook.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--labels", nargs="+", default=["1", "0"])
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--classification-max-output-tokens", type=int, default=50000)
    parser.add_argument("--codebook-max-output-tokens", type=int, default=50000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_response_text(body: dict) -> str:
    text_parts = []

    for item in body.get("output", []):
        if item.get("type") != "message":
            continue

        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text_parts.append(content["text"])

    if not text_parts:
        raise ValueError(f"Response did not contain output text. Status: {body.get('status')!r}")

    return "".join(text_parts)

def clean_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1]

def send_request(api_key: str, request_body: dict, max_retries: int) -> dict:
    for attempt in range(max_retries + 1):
        response = requests.post(
            RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
            timeout=600,
        )

        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_retries:
            break

        try:
            delay = float(response.headers.get("Retry-After"))
        except (TypeError, ValueError):
            delay = 30

        print(f"retrying in {delay} seconds ({attempt + 1}/{max_retries})")
        time.sleep(delay)

    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text}")

    body = response.json()

    if body.get("error"):
        raise RuntimeError(f"OpenRouter response error: {body['error']}")

    if body.get("status") == "incomplete":
        raise RuntimeError(f"Response was incomplete: {body.get('incomplete_details')}")

    return body


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    classification_prompt = args.classification_prompt_path.read_text(encoding="utf-8")
    codebook_prompt = args.codebook_prompt_path.read_text(encoding="utf-8")
    allowed_labels = set(args.labels)

    documents = pl.read_csv(args.data_path)
    document_ids = documents.get_column(args.id_column).cast(pl.String).to_list()

    args.out_folder.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(args.model)
    output_classifications_path = args.classifications_path or args.out_folder / f"{model_name}_initial_classifications.json"
    errors_path = args.out_folder / f"{model_name}_classification_errors.json"

    if args.classifications_path is not None:
        classifications = json.loads(args.classifications_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(classifications)} completed classifications from {args.classifications_path}")
    else:
        classifications = []

    completed_by_id = {str(record["document_id"]): record for record in classifications}
    errors = []

    for position, document in enumerate(documents.iter_rows(named=True), start=1):
        document_id = str(document[args.id_column])
        document_text = str(document[args.text_column])

        if document_id in completed_by_id:
            print(f"{position}/{documents.height}: skipping completed document {document_id}")
            continue

        print(f"{position}/{documents.height}: classifying {document_id}")

        try:
            response_body = send_request(
                api_key,
                {
                    "model": args.model,
                    "instructions": classification_prompt,
                    "input": f"Document:\n\n{document_text}",
                    "temperature": 0,
                    "max_output_tokens": args.classification_max_output_tokens,
                },
                args.max_retries,
            )

            output_text = clean_json(get_response_text(response_body))
            classification = json.loads(output_text)
            label = str(classification.get("label"))
            decision_basis = classification.get("decision_basis")
            confidence = classification.get("confidence")

            if label not in allowed_labels:
                raise ValueError(f"Invalid label: {label!r}")

            if not decision_basis:
                raise ValueError("Response was missing decision_basis.")

            record = {
                "document_id": document_id,
                "text": document_text,
                "label": label,
                "decision_basis": decision_basis,
                "confidence": confidence,
            }

            completed_by_id[document_id] = record
            classifications = [completed_by_id[doc_id] for doc_id in document_ids if doc_id in completed_by_id]

            output_classifications_path.write_text(
                json.dumps(classifications, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"  -> {label}: {decision_basis}")

        except Exception as error:
            errors.append({
                "document_id": document_id,
                "text": document_text,
                "error": f"{type(error).__name__}: {error}",
            })

            errors_path.write_text(
                json.dumps(errors, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"ERROR {len(errors)}/{args.max_errors}: {error}")

            if len(errors) >= args.max_errors:
                raise RuntimeError(
                    f"Stopped after {len(errors)} errors. Resume with "
                    f'--classifications-path "{output_classifications_path}"'
                )
            
    classifications = [completed_by_id[doc_id] for doc_id in document_ids if doc_id in completed_by_id]

    if errors:
        print(f"Completed with {len(errors)} errors: {errors_path}")

    args.out_folder.mkdir(parents=True, exist_ok=True)
    model_name = safe_name(args.model)
    output_classifications_path = args.out_folder / f"{model_name}_initial_classifications.json"
    codebook_path = args.out_folder / f"{model_name}_codebook.json"

    codebook_input = {
        "original_classification_instructions": classification_prompt,
        "classification_records": [
            {
                "text": record["text"],
                "label": record["label"],
                "decision_basis": record["decision_basis"],
                "confidence": record["confidence"]
            } for record in classifications
        ]
    }

    print(f"\nGenerating codebook with {args.model}...")

    codebook_response = send_request(
        api_key,
        {
            "model": args.model,
            "instructions": codebook_prompt,
            "input": json.dumps(codebook_input, ensure_ascii=False, indent=2),
            "temperature": 0,
            "max_output_tokens": args.codebook_max_output_tokens,
        },
        args.max_retries,
    )

    codebook_text = clean_json(get_response_text(codebook_response))
    codebook = json.loads(codebook_text)

    codebook_path.write_text(
        json.dumps(codebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Classifications:",
        args.classifications_path
        or output_classifications_path,
    )
    print("Codebook:", codebook_path)


if __name__ == "__main__":
    main()
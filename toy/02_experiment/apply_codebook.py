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
    parser.add_argument("--codebook-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/populism_codebookapply.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/apply_codebook.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--labels", nargs="+", default=["1", "0"])
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=100000)
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


def clean_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1]


def send_request(api_key: str, request_body: dict, max_retries: int) -> dict:
    for attempt in range(max_retries + 1):
        response = requests.post(
            RESPONSES_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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

    documents = pl.read_csv(args.data_path)

    codebook = json.loads(args.codebook_path.read_text(encoding="utf-8"))
    codebook_text = json.dumps(codebook, ensure_ascii=False, indent=2)
    application_prompt = args.prompt_path.read_text(encoding="utf-8").strip()

    instructions = (
        f"{application_prompt}\n\n"
        "The codebook is below, formatted as a JSON object:\n\n"
        f"{codebook_text}"
    )

    labels = []
    decision_explanations = []
    confidences = []
    errors = []

    for position, document in enumerate(documents.iter_rows(named=True), start=1):
        document_id = str(document[args.id_column])
        document_text = str(document[args.text_column])

        print(f"{position}/{documents.height}: labeling {document_id}")

        model_input = {
            "codebook": codebook,
            "document": {
                "document_id": document_id,
                "text": document_text,
            },
        }

        try:
            response_body = send_request(
                api_key,
                {
                    "model": args.model,
                    "instructions": instructions,
                    "input": json.dumps(model_input, ensure_ascii=False, indent=2),
                    "temperature": 0,
                    "max_output_tokens": args.max_output_tokens,
                },
                args.max_retries,
            )

            output_text = clean_json(get_response_text(response_body))
            classification = json.loads(output_text)
            label = str(classification.get("label"))
            decision_basis = classification.get("decision_basis")
            confidence = classification.get("confidence")

            if label not in args.labels:
                raise ValueError(f"Unexpected label: {label!r}")

            if not decision_basis:
                raise ValueError("Response is missing decision_basis.")

            labels.append(label)
            decision_explanations.append(decision_basis)
            confidences.append(confidence)
            errors.append(None)

            print(f"{label}: {decision_basis}")

        except Exception as error:
            labels.append(None)
            decision_explanations.append(None)
            confidences.append(None)
            errors.append(f"{type(error).__name__}: {error}")

            print(f"ERROR: {error}")

    results_df = documents.with_columns(
        pl.Series("label", labels, dtype=pl.String, strict=False),
        pl.Series("decision_basis", decision_explanations, dtype=pl.String, strict=False),
        pl.Series("confidence", confidences, dtype=pl.Int64, strict=False),
        pl.Series("error", errors, dtype=pl.String, strict=False),
    )

    args.out_folder.mkdir(parents=True, exist_ok=True)

    model_name = safe_name(args.model)
    codebook_name = safe_name(args.codebook_path.stem)
    output_name = f"model_{model_name}_codebook_{codebook_name}_labels"
    csv_path = args.out_folder / f"{output_name}.csv"

    results_df.write_csv(csv_path)

    print("Output:", csv_path)
    print("Successful:", results_df.filter(pl.col("error").is_null()).height)
    print("Errors:", results_df.filter(pl.col("error").is_not_null()).height)


if __name__ == "__main__":
    main()
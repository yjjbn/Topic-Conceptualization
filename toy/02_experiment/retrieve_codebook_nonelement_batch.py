import argparse
import json
import os
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
TERMINAL_FAILURE_STATUSES = {"failed", "expired", "cancelled"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Override the default output path ending in _rules.csv.",
    )
    parser.add_argument(
        "--json-output-path",
        type=Path,
        help="Override the companion JSON output path ending in _rules.json.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def clean_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in response: {text!r}")
    return text[start : end + 1]


def get_batch(api_key, batch_id, max_retries):
    for attempt in range(max_retries + 1):
        response = requests.get(
            f"{BATCHES_URL}/{batch_id}",
            headers={"Authorization": f"Bearer {api_key}"},
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


def default_output_path(meta):
    original = Path(meta["output_path"])
    return original.with_name(f"{original.stem}_rules{original.suffix}")


def default_json_output_path(csv_output_path):
    return csv_output_path.with_suffix(".json")


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    batch = get_batch(api_key, meta["batch_id"], args.max_retries)
    status = batch.get("status")
    counts = batch.get("request_counts", {})
    print("Status:", status)
    print(
        f"Requests: {counts.get('completed', 0)} completed, "
        f"{counts.get('failed', 0)} failed, "
        f"{counts.get('total', 0)} total"
    )

    if status in TERMINAL_FAILURE_STATUSES:
        raise RuntimeError(f"Batch ended with status {status!r}.")
    if status != "completed" or batch.get("results") is None:
        raise RuntimeError("Batch is not complete yet. Run this script again later.")

    documents = pl.read_csv(meta["data_path"])
    submitted = {
        record["custom_id"]: record for record in meta["submitted_documents"]
    }
    results_by_id = {}
    json_results_by_id = {}

    for result in batch["results"]:
        custom_id = result.get("custom_id")
        submitted_record = submitted.get(custom_id)
        if submitted_record is None:
            print(f"Skipping unknown custom_id: {custom_id!r}")
            continue
        document_id = str(submitted_record["document_id"])
        raw_response = None

        try:
            if result.get("error"):
                raise RuntimeError(result["error"])

            response = result.get("response")
            if not response:
                raise ValueError("Result contained no response.")
            if response.get("status_code") != 200:
                raise RuntimeError(
                    f"HTTP {response.get('status_code')}: {response.get('body')}"
                )

            raw_response = response["body"]["choices"][0]["message"]["content"]
            classification = json.loads(clean_json(raw_response))

            label = str(classification.get("label"))
            if label not in meta["labels"]:
                raise ValueError(f"Unexpected label: {label!r}")

            codebook_rule = classification.get("codebook_rule")
            explanation = classification.get("explanation")
            if not isinstance(codebook_rule, list) or not all(
                isinstance(rule, str) for rule in codebook_rule
            ):
                raise ValueError("codebook_rule must be an array of strings.")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError("explanation must be a non-empty string.")

            results_by_id[document_id] = {
                "label": label,
                "codebook_rule": json.dumps(
                    codebook_rule, ensure_ascii=False, separators=(",", ":")
                ),
                "explanation": explanation,
                "error": None,
            }
            json_results_by_id[document_id] = {
                "document_id": document_id,
                "label": label,
                "codebook_rule": codebook_rule,
                "explanation": explanation,
                "error": None,
            }

        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            results_by_id[document_id] = {
                "label": None,
                "codebook_rule": None,
                "explanation": None,
                "error": error_message,
            }
            json_results_by_id[document_id] = {
                "document_id": document_id,
                "label": None,
                "codebook_rule": None,
                "explanation": None,
                "error": error_message,
                "raw_response": raw_response,
            }

    ids = documents.get_column(meta["id_column"]).cast(pl.String).to_list()
    fields = ["label", "codebook_rule", "explanation", "error"]
    output_columns = {
        field: [
            results_by_id.get(document_id, {}).get(
                field,
                "Missing batch result" if field == "error" else None,
            )
            for document_id in ids
        ]
        for field in fields
    }

    results_df = documents.with_columns(
        *[
            pl.Series(field, values, dtype=pl.String, strict=False)
            for field, values in output_columns.items()
        ]
    )

    output_path = args.output_path or default_output_path(meta)
    json_output_path = (
        args.json_output_path or default_json_output_path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)
    json_results = [
        json_results_by_id.get(
            document_id,
            {
                "document_id": document_id,
                "label": None,
                "codebook_rule": None,
                "explanation": None,
                "error": "Missing batch result",
            },
        )
        for document_id in ids
    ]
    json_output_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Saved CSV:", output_path)
    print("Saved JSON:", json_output_path)


if __name__ == "__main__":
    main()

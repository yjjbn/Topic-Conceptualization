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
        help="Override the default output path ending in _element.csv.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def clean_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON in response: {text!r}")
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


def json_string(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def default_output_path(meta):
    original = Path(meta["output_path"])
    return original.with_name(f"{original.stem}_element{original.suffix}")


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
    submitted = {record["custom_id"]: record for record in meta["submitted_documents"]}
    results_by_id = {}

    for result in batch["results"]:
        custom_id = result.get("custom_id")
        submitted_record = submitted.get(custom_id)
        if submitted_record is None:
            print(f"Skipping unknown custom_id: {custom_id!r}")
            continue
        document_id = str(submitted_record["document_id"])

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

            text = response["body"]["choices"][0]["message"]["content"]
            classification = json.loads(clean_json(text))

            response_document_id = classification.get("document_id")
            if response_document_id is not None and str(response_document_id) != document_id:
                raise ValueError(
                    f"Response document_id {response_document_id!r} did not match "
                    f"submitted document_id {document_id!r}."
                )

            label = str(classification.get("label"))
            if label not in meta["labels"]:
                raise ValueError(f"Unexpected label: {label!r}")

            element_findings = classification.get("element_findings")
            decision_basis = classification.get("decision_basis")
            application_tips_used = classification.get("application_tips_used")
            rationale = classification.get("rationale")

            if not isinstance(element_findings, list):
                raise ValueError("element_findings must be a JSON array.")
            if not isinstance(decision_basis, dict):
                raise ValueError("decision_basis must be a JSON object.")
            if not isinstance(application_tips_used, list):
                raise ValueError("application_tips_used must be a JSON array.")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("rationale must be a non-empty string.")

            results_by_id[document_id] = {
                "label": label,
                "element_findings": json_string(element_findings),
                "decision_basis": json_string(decision_basis),
                "application_tips_used": json_string(application_tips_used),
                "rationale": rationale,
                "error": None,
            }

        except Exception as error:
            results_by_id[document_id] = {
                "label": None,
                "element_findings": None,
                "decision_basis": None,
                "application_tips_used": None,
                "rationale": None,
                "error": f"{type(error).__name__}: {error}",
            }

    ids = documents.get_column(meta["id_column"]).cast(pl.String).to_list()
    fields = [
        "label",
        "element_findings",
        "decision_basis",
        "application_tips_used",
        "rationale",
        "error",
    ]
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

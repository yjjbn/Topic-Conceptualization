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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def clean_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON in response: {text!r}")
    return text[start:end + 1]


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


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    batch = get_batch(api_key, meta["batch_id"], args.max_retries)

    documents = pl.read_csv(meta["data_path"])
    submitted = {x["custom_id"]: x for x in meta["submitted_documents"]}
    results_by_id = {}

    for result in batch["results"]:
        custom_id = result["custom_id"]
        document_id = str(submitted[custom_id]["document_id"])

        try:
            if result.get("error"):
                raise RuntimeError(result["error"])

            response = result["response"]
            if response["status_code"] != 200:
                raise RuntimeError(f"HTTP {response['status_code']}: {response.get('body')}")

            text = response["body"]["choices"][0]["message"]["content"]
            classification = json.loads(clean_json(text))

            label = str(classification.get("label"))
            if label not in meta["labels"]:
                raise ValueError(f"Unexpected label: {label!r}")

            results_by_id[document_id] = {
                "label": label,
                "decision_basis": classification.get("decision_basis"),
                "confidence": classification.get("confidence"),
                "error": None,
            }

        except Exception as error:
            results_by_id[document_id] = {
                "label": None,
                "decision_basis": None,
                "confidence": None,
                "error": f"{type(error).__name__}: {error}",
            }

    ids = documents.get_column(meta["id_column"]).cast(pl.String).to_list()

    labels = [results_by_id.get(i, {}).get("label") for i in ids]
    decision_basis = [results_by_id.get(i, {}).get("decision_basis") for i in ids]
    confidence = [results_by_id.get(i, {}).get("confidence") for i in ids]
    errors = [results_by_id.get(i, {}).get("error", "Missing batch result") for i in ids]

    results_df = documents.with_columns(
        pl.Series("label", labels, dtype=pl.String, strict=False),
        pl.Series("decision_basis", decision_basis, dtype=pl.String, strict=False),
        pl.Series("confidence", confidence, strict=False),
        pl.Series("error", errors, dtype=pl.String, strict=False),
    )

    output_path = Path(meta["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)

if __name__ == "__main__":
    main()
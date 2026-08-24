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
    parser.add_argument(
        "--output-format",
        choices=("csv", "json"),
        default="csv",
        help="Save a joined CSV or the raw model-output JSON (default: csv).",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def clean_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON in response: {text!r}")
    return text[start:end + 1]


def csv_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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
    output_fields = []

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
            if not isinstance(classification, dict):
                raise ValueError("The model output must be a JSON object.")

            for field in classification:
                if field not in output_fields:
                    output_fields.append(field)

            results_by_id[document_id] = {
                "values": classification,
                "error": None,
            }

        except Exception as error:
            results_by_id[document_id] = {
                "values": {},
                "error": f"{type(error).__name__}: {error}",
            }

    ids = documents.get_column(meta["id_column"]).cast(pl.String).to_list()

    errors = [results_by_id.get(i, {}).get("error", "Missing batch result") for i in ids]
    output_path = Path(meta["output_path"]).with_suffix(f".{args.output_format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "json":
        raw_outputs = [
            results_by_id[document_id]["values"]
            for document_id in ids
            if document_id in results_by_id
            and results_by_id[document_id]["error"] is None
        ]

        output_path.write_text(
            json.dumps(raw_outputs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Results:", output_path)
        return

    output_columns = [
        pl.Series(
            field,
            [
                csv_value(results_by_id.get(i, {}).get("values", {}).get(field))
                for i in ids
            ],
            dtype=pl.String,
        )
        for field in output_fields
    ]

    results_df = documents.with_columns(
        *output_columns,
        pl.Series("_retrieval_error", errors, dtype=pl.String, strict=False),
    )

    results_df.write_csv(output_path)
    print("Results:", output_path)

if __name__ == "__main__":
    main()

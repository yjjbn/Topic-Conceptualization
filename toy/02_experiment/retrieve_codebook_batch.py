import argparse
import json
import os
import time
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

BATCHES_URL = "https://openrouter.ai/api/beta/batches"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=Path, required=True)
    return parser.parse_args()


def clean_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Could not find a JSON array in response: {text!r}")
    return text[start : end + 1]


def csv_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_batch(api_key, batch_id, max_retries):
    response = requests.get(
        f"{BATCHES_URL}/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600,
    )
    return response.json()


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    batch = get_batch(api_key, meta["batch_id"], args.max_retries)

    documents = pl.read_csv(meta["data_path"])
    submitted_requests = meta["submitted_requests"]

    submitted = {item["custom_id"]: item for item in submitted_requests}
    results_by_id = {}
    output_fields = []
    raw_outputs_by_custom_id = {}

    for result in batch["results"]:
        custom_id = result["custom_id"]
        request_documents = submitted[custom_id]["documents"]
        expected_ids = {str(document["document_id"]) for document in request_documents}

        try:
            if result.get("error"):
                raise RuntimeError(result["error"])

            response = result["response"]
            if response["status_code"] != 200:
                raise RuntimeError(f"HTTP {response['status_code']}: {response.get('body')}")

            text = response["body"]["choices"][0]["message"]["content"]
            model_output = json.loads(clean_json(text))
            raw_outputs_by_custom_id[custom_id] = model_output
            if not isinstance(model_output, list):
                raise ValueError(
                    "The model output must be a JSON array."
                )
            classifications = model_output
            if not all(isinstance(item, dict) for item in classifications):
                raise ValueError("Every result must be a JSON object.")

            request_results = {}
            for classification in classifications:
                document_id = str(classification["document_id"])
                if document_id not in expected_ids:
                    raise ValueError(f"Unexpected document ID: {document_id!r}")

                for field in classification:
                    if field not in output_fields:
                        output_fields.append(field)

                request_results[document_id] = classification

            for document_id in expected_ids:
                if document_id in request_results:
                    results_by_id[document_id] = {
                        "values": request_results[document_id],
                        "error": None,
                    }
                else:
                    results_by_id[document_id] = {
                        "values": {},
                        "error": "Missing document result in model response",
                    }

        except Exception as error:
            for document_id in expected_ids:
                results_by_id[document_id] = {
                    "values": {},
                    "error": f"{type(error).__name__}: {error}",
                }

    ids = documents.get_column(meta["id_column"]).cast(pl.String).to_list()

    errors = [results_by_id.get(i, {}).get("error", "Missing batch result") for i in ids]
    output_path = Path(meta["output_path"])
    raw_output_path = output_path.with_name(
        f"{output_path.stem.removesuffix('_results')}_raw_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_outputs = [
        model_result
        for item in submitted_requests
        for model_result in raw_outputs_by_custom_id.get(item["custom_id"], [])
    ]
    raw_output_path.write_text(
        json.dumps(raw_outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Raw results:", raw_output_path)

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
    print("CSV results:", output_path)

if __name__ == "__main__":
    main()

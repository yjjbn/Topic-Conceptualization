import argparse
import json
import os
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv

from apply_codebook_batch import Classification

BATCHES_URL = "https://openrouter.ai/api/beta/batches"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", help="batch ID to retrieve")
    args = parser.parse_args()
    try:
        args.meta_path = find_meta_path(
            Path(__file__).resolve().parent / "out", args.batch_id
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def find_meta_path(search_dir: Path, batch_id: str) -> Path:
    files = [
        file for file in sorted(search_dir.rglob("batch_meta_*.json"))
        if json.loads(file.read_text(encoding="utf-8")).get("batch_id") == batch_id
    ]
    if not files:
        raise ValueError(f"No matching batch metadata found under {search_dir}.")
    if len(files) != 1:
        raise ValueError(f"Multiple metadata files contain batch ID {batch_id!r}.")
    return files[0]


def clean_json(text):
    text = text.strip()
    starts = [index for char in ("{", "[") if (index := text.find(char)) != -1]
    start = min(starts) if starts else -1
    end = text.rfind("}" if start != -1 and text[start] == "{" else "]")
    if start == -1 or end == -1:
        raise ValueError("Could not find a JSON object or array in response.")
    return text[start : end + 1]


def csv_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_batch(api_key, batch_id):
    response = requests.get(
        f"{BATCHES_URL}/{batch_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    print("Retrieving batch:", meta["batch_id"])
    print("Meta:", args.meta_path)
    batch = get_batch(api_key, meta["batch_id"])

    submitted_requests = meta["submitted_requests"]
    submitted_documents = [
        document
        for request in submitted_requests
        for document in request["documents"]
    ]
    # Use the submitted snapshot, not the current source CSV or returned IDs.
    documents = pl.DataFrame(
        submitted_documents, schema={"document_id": pl.String, "text": pl.String}
    )
    ids = documents.get_column("document_id").to_list()
    if len(set(ids)) != len(ids):
        raise ValueError("Submission metadata contains duplicate document IDs.")

    submitted = {item["custom_id"]: item for item in submitted_requests}
    results_by_id = {}
    output_fields = [field for field in Classification.model_fields if field != "document_id"]

    for result in batch.get("results") or []:
        custom_id = result["custom_id"]
        if custom_id not in submitted:
            print(f"WARNING: Ignoring unknown request {custom_id!r}.")
            continue
        expected_ids = {
            str(document["document_id"])
            for document in submitted[custom_id]["documents"]
        }

        try:
            if result.get("error"):
                raise RuntimeError(result["error"])
            response = result["response"]
            if response["status_code"] != 200:
                raise RuntimeError(f"HTTP {response['status_code']}: {response.get('body')}")
            text = response["body"]["choices"][0]["message"]["content"]
            model_output = json.loads(clean_json(text))
            if isinstance(model_output, dict) and isinstance(model_output.get("classifications"), list):
                model_output = model_output["classifications"]
            if isinstance(model_output, dict):
                entries = list(model_output.items())
            elif isinstance(model_output, list):
                entries = []
                for item in model_output:
                    if not isinstance(item, dict) or "document_id" not in item:
                        print(f"WARNING: Ignoring a result without a document ID in {custom_id!r}.")
                        continue
                    entries.append((str(item["document_id"]), item))
            else:
                raise ValueError("Expected an ID-keyed object or classifications array.")

            request_results = {}
            for document_id, classification in entries:
                if document_id not in expected_ids:
                    print(f"WARNING: Ignoring unexpected document ID {document_id!r}.")
                    continue
                if document_id in request_results:
                    print(f"WARNING: Duplicate result for {document_id!r}; keeping the first.")
                    continue
                try:
                    validated = Classification.model_validate(classification)
                    if validated.document_id != document_id:
                        raise ValueError(f"Document ID does not match key: {document_id!r}")
                    request_results[document_id] = {
                        "values": validated.model_dump(), "error": None
                    }
                except (TypeError, ValueError) as error:
                    request_results[document_id] = {
                        "values": {}, "error": f"{type(error).__name__}: {error}"
                    }

            for document_id in expected_ids:
                results_by_id[document_id] = request_results.get(document_id, {
                    "values": {}, "error": "Missing document result in model response"
                })
        except Exception as error:
            for document_id in expected_ids:
                results_by_id[document_id] = {
                    "values": {}, "error": f"{type(error).__name__}: {error}"
                }

    errors = [results_by_id.get(i, {}).get("error", "Missing batch result") for i in ids]
    # Current metadata has no output_path; save beside the metadata file.
    file_prefix = args.meta_path.stem.removeprefix("batch_meta_")
    output_path = (
        Path(meta["output_path"]) if meta.get("output_path")
        else args.meta_path.with_name(f"results_{file_prefix}.csv")
    )
    raw_output_path = output_path.with_name(f"{output_path.stem}_raw_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Raw batch response:", raw_output_path)

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
    missing_count = sum(error is not None for error in errors)
    if missing_count:
        print(f"WARNING: {missing_count}/{len(ids)} submitted documents have missing or invalid results.")
    print(f"CSV results: {output_path} ({len(ids)} submitted documents)")

if __name__ == "__main__":
    main()

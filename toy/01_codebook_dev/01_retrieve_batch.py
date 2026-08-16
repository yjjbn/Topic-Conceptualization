import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
TERMINAL_FAILURE_STATUSES = {"failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def get_batch(api_key: str, batch_id: str, max_retries: int) -> dict:
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


def get_response_text(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"Response did not contain chat completion text: {body}")


def clean_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in response: {text!r}")

    return text[start:end + 1]


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))

    batch_id = meta["batch_id"]
    allowed_labels = set(meta["labels"])
    all_document_ids = [str(x) for x in meta["all_document_ids"]]
    submitted_by_custom_id = {
        record["custom_id"]: record
        for record in meta["submitted_documents"]
    }

    classifications_path = Path(meta["classifications_path"])
    errors_path = classifications_path.with_name(
        f"{classifications_path.stem}_errors.json"
    )

    print(f"Retrieving batch {batch_id}...")

    batch = get_batch(api_key, batch_id, args.max_retries)

    status = batch.get("status")
    counts = batch.get("request_counts", {})

    print("Status:", status)
    print(
        f"Requests: {counts.get('completed', 0)} completed, "
        f"{counts.get('failed', 0)} failed, "
        f"{counts.get('total', 0)} total"
    )

    results = batch.get("results")

    # Load classifications that existed before this batch, if any.
    if classifications_path.exists():
        classifications = json.loads(
            classifications_path.read_text(encoding="utf-8")
        )
    else:
        classifications = []

    completed_by_id = {
        str(record["document_id"]): record
        for record in classifications
    }

    errors = []

    for result in results:
        custom_id = result.get("custom_id")

        try:
            if custom_id not in submitted_by_custom_id:
                raise ValueError(f"Unknown custom_id: {custom_id!r}")

            submitted = submitted_by_custom_id[custom_id]
            document_id = str(submitted["document_id"])
            document_text = submitted["text"]

            if result.get("error"):
                raise RuntimeError(f"Batch request error: {result['error']}")

            response = result.get("response")
            if not response:
                raise ValueError("Result contained neither response nor error.")

            if response.get("status_code") != 200:
                raise RuntimeError(
                    f"Request returned HTTP {response.get('status_code')}: "
                    f"{response.get('body')}"
                )

            response_body = response["body"]
            output_text = clean_json(get_response_text(response_body))
            classification = json.loads(output_text)

            label = str(classification.get("label"))
            decision_basis = classification.get("decision_basis")
            confidence = classification.get("confidence")

            if label not in allowed_labels:
                raise ValueError(f"Invalid label: {label!r}")

            if not decision_basis:
                raise ValueError("Response was missing decision_basis.")

            completed_by_id[document_id] = {
                "document_id": document_id,
                "text": document_text,
                "label": label,
                "decision_basis": decision_basis,
                "confidence": confidence,
            }

        except Exception as error:
            errors.append({
                "custom_id": custom_id,
                "document_id": submitted_by_custom_id.get(custom_id, {}).get("document_id"),
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"ERROR {custom_id}: {error}")

    # Restore original dataset ordering.
    classifications = [
        completed_by_id[document_id]
        for document_id in all_document_ids
        if document_id in completed_by_id
    ]

    classifications_path.parent.mkdir(parents=True, exist_ok=True)
    classifications_path.write_text(
        json.dumps(classifications, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if errors:
        errors_path.write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"\nSaved {len(classifications)} classifications:")
    print(classifications_path)

if __name__ == "__main__":
    main()
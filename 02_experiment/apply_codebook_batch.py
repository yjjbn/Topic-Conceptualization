import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

BATCHES_URL = "https://openrouter.ai/api/beta/batches"
DOCUMENTS_PER_REQUEST = 40


class StrictModel(BaseModel, extra="forbid", strict=True):
    pass

# class Classification(StrictModel):
#     document_id: str
#     label: Literal["0", "1"]
#     codebook_rule: list[str]
#     explanation: str

class Classification(StrictModel):
    document_id: str
    label: Literal["0", "1"]
    explanation: str


class GoogleClassifications(StrictModel):
    classifications: list[Classification]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nametag", default="explanation_codebookLLM")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--codebook-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookapply_300.csv"))
    parser.add_argument("--data-encoding", default="utf8", help="CSV encoding; utf8-lossy replaces invalid bytes")
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/apply_codebookLLM_grimmer.txt"))
    parser.add_argument("--dry-run", action="store_true", help="validate and list batches without submitting")
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--test-rows", type=int, default=None)
    parser.add_argument("--note", default="", help="add a note if you want")
    # parser.add_argument("--max-output-tokens", type=int, default=5000)
    args = parser.parse_args()
    return args


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def build_response_model(document_ids: list[str]) -> type[BaseModel]:
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    fields = {}
    for index, document_id in enumerate(document_ids):
        classification_model = create_model(
            f"Classification_{index}",
            __base__=Classification,
            document_id=(Literal[document_id], ...),
        )
        # Aliases support IDs that are not valid Python field names.
        fields[f"document_{index}"] = (
            classification_model, Field(alias=document_id)
        )
    return create_model("CodebookClassifications", __base__=StrictModel, **fields)


def build_response_format(document_ids: list[str], model: str = "") -> dict:
    response_model = (
        GoogleClassifications if model.lower().startswith("google/")
        else build_response_model(document_ids)
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "codebook_classifications",
            "strict": True,
            "schema": response_model.model_json_schema(by_alias=True),
        },
    }


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
    result = response.json()
    if not response.ok or result.get("error"):
        error = result.get("error", result)
        raise RuntimeError(f"Batch API request failed (HTTP {response.status_code}): {error}")
    return result


def prepare_batch(args):

    documents = pl.read_csv(
        args.data_path, schema_overrides={"document_id": pl.String},
        encoding=getattr(args, "data_encoding", "utf8"),
    )
    if args.test_rows is not None:
        documents = documents.sample(n=args.test_rows, seed=1)

    codebook = json.loads(args.codebook_path.read_text(encoding="utf-8"))
    application_prompt = args.prompt_path.read_text(encoding="utf-8")
    # if args.model.lower().startswith("google/"):
    #     # Replace the ID-keyed output instructions while retaining the task.
    #     application_prompt = application_prompt.split("## OUTPUT FORMAT", 1)[0]
    #     application_prompt += (
    #         '\n\n## OUTPUT FORMAT\nReturn a JSON object matching the supplied schema'
    #         "Include every input document ID exactly once, with no additional IDs. "
    #     )

    instructions = (
        f"{application_prompt}\n\n"
        "The codebook is provided below, formatted as a JSON object.\n\nApply the codebook independently to every document.\n\n"
        f"{json.dumps(codebook, indent=2)}"
    )

    document_inputs = [
        {
            "document_id": str(document["document_id"]),
            "text": str(document["text"]),
        }
        for document in documents.iter_rows(named=True)
    ]
    if len({item["document_id"] for item in document_inputs}) != len(document_inputs):
        raise ValueError("Document IDs must be unique.")
    if not document_inputs:
        raise ValueError("No documents to classify.")
    requests_list, submitted_requests = [], []

    for start in range(0, len(document_inputs), DOCUMENTS_PER_REQUEST):
        request_documents = document_inputs[start : start + DOCUMENTS_PER_REQUEST]
        request_number = start // DOCUMENTS_PER_REQUEST + 1
        custom_id = f"request-{request_number:08d}"

        requests_list.append({
            "custom_id": custom_id,
            "body": {
                "response_format": build_response_format(
                    [document["document_id"] for document in request_documents], args.model
                ),
                "provider": {"require_parameters": True},
                "plugins": [{"id": "response-healing"}],
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(request_documents, ensure_ascii=False),
                    },
                ],
                # "temperature": 0,
                # "max_completion_tokens": args.max_output_tokens
            },
        })

        submitted_requests.append({
            "custom_id": custom_id,
            "document_ids": [document["document_id"] for document in request_documents],
            "documents": request_documents,
        })

    print(
        f"Prepared {args.model}, run {args.run}: {len(requests_list)} requests for "
        f"{len(document_inputs)} documents."
    )

    payload = {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "requests": requests_list,
    }

    return payload, submitted_requests, instructions


def submit_batch(args, prepared, api_key, source=None):
    payload, submitted_requests, instructions = prepared
    submitted_at = datetime.now(timezone.utc).isoformat()
    submission = request_json("POST", BATCHES_URL, api_key, payload)
    batch_id = submission.get("id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise RuntimeError(f"Batch submission returned no batch ID: {submission}")

    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag)
    codebook_name = safe_name(args.codebook_path.stem)
    output_folder = (
        args.out_folder
        / f"{nametag}"
        / f"run_{args.run}"
        / f"{model_name}"
        / f"{codebook_name}"
        / f"{batch_id}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    if args.test_rows is None:
        file_prefix = f"{model_name}_{nametag}_run_{args.run}"
    else:
        file_prefix = f"test_{args.test_rows}_{model_name}_{nametag}_run_{args.run}"

    meta_path = output_folder / f"batch_meta_{file_prefix}.json"

    note_text = args.note or ""
    prompt = "\n\nSystem instructions:\n" + instructions + "\n"

    meta = {
        "batch_id": batch_id,
        "note": note_text,
        "submitted_at": submitted_at,
        "batches_url": BATCHES_URL,
        "endpoint": payload["endpoint"],
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
            if name != "original_request_documents"
        },
        "documents_per_request": DOCUMENTS_PER_REQUEST,
        "document_count": sum(len(item["documents"]) for item in submitted_requests),
        "request_count": len(submitted_requests),
        "source": source,
        "submitted_requests": submitted_requests,
        "submission": submission,
        "prompt": prompt,
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Batch ID:", batch_id)
    print("Status:", submission.get("status"))
    print("Meta:", meta_path)
    print(
        f'\nRetrieve later with:\nuv run python retrieve_codebook_batch.py '
        f'--meta-path "{meta_path}"'
    )


def main():
    args = parse_args()
    prepared = prepare_batch(args)
    if args.dry_run:
        print("Dry run complete; no API requests submitted.")
        return
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    submit_batch(args, prepared, api_key)


if __name__ == "__main__":
    main()

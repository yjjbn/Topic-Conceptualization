"""Apply one codebook repeatedly, with batch submission and retrieval in this file.

From this folder:
  uv run apply_batch.py submit --model MODEL --data-path DATA.csv --codebook-path CODEBOOK.json --prompt-path PROMPT.txt --repeats 50
  uv run apply_batch.py retrieve --meta-path out/TAG/MODEL/CODEBOOK/BATCH_ID/batch_meta.json

Each run covers the dataset in groups of 40 documents by default.
Use --mode streaming to send regular requests sequentially and export automatically.
Streaming results use a streaming_run_TIMESTAMP folder and the same metadata files.
Model parameters come from ../01_codebook_dev/model_parameters.toml.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Literal

import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from anthropic import transform_schema

BATCHES_URL = "https://openrouter.ai/api/v1/batches"
PARAMETERS = Path(__file__).resolve().parents[1] / "01_codebook_dev/model_parameters.toml"


class Classification(BaseModel, extra="forbid", strict=True):
    document_id: str
    label: Literal["0", "1"]
    explanation: str


class Classifications(BaseModel, extra="forbid", strict=True):
    classifications: list[Classification]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--model", required=True)
    submit.add_argument("--mode", choices=["batch", "streaming"], default="batch")
    submit.add_argument("--provider", help="use only this OpenRouter provider, in either mode")
    submit.add_argument("--session-id", help="session ID sent with each streaming request; ignored in batch mode")
    submit.add_argument("--data-path", type=Path, required=True)
    submit.add_argument("--codebook-path", type=Path, required=True)
    submit.add_argument("--prompt-path", type=Path, required=True)
    submit.add_argument("--repeats", type=int, default=50)
    submit.add_argument("--start-run", "--run", type=int, default=1)
    submit.add_argument("--runs", type=int, nargs="+", help="specific run numbers; overrides repeats/start-run")
    submit.add_argument("--documents-per-request", type=int, default=40)
    submit.add_argument("--nametag", default="")
    submit.add_argument("--out-folder", type=Path, default=Path("./out"))
    submit.add_argument("--dry-run", action="store_true", help="save requests without sending them")
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--meta-path", type=Path, required=True)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api_request(method, api_key, batch_id="", payload=None):
    response = requests.request(
        method, BATCHES_URL + (f"/{batch_id}" if batch_id else ""),
        headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=(30, 600),
    )
    response.raise_for_status()
    return response.json()


def prepare_batch(args):
    data = pl.read_csv(args.data_path, schema_overrides={"document_id": pl.String})
    documents = data.select("document_id", "text").to_dicts()
    ids = data["document_id"].to_list()
    if not documents or len(set(ids)) != len(ids) or any(
        doc["document_id"] is None or doc["text"] is None for doc in documents
    ):
        raise ValueError("Documents need unique, non-null IDs and non-null text.")
    runs = args.runs or list(range(args.start_run, args.repeats))
    if not runs or min(runs) < 1 or len(set(runs)) != len(runs) or args.documents_per_request < 1:
        raise ValueError("Use positive, unique run numbers and a positive documents-per-request value.")
    codebook = json.loads(args.codebook_path.read_text(encoding="utf-8"))
    prompt = args.prompt_path.read_text(encoding="utf-8")
    prompt += "\n\nCodebook:\n" + json.dumps(codebook, ensure_ascii=False, indent=2)
    with PARAMETERS.open("rb") as file:
        parameters = tomllib.load(file)[args.model.split(":", 1)[0]]
    provider = parameters.pop("provider", {})
    limit = parameters.get("max_completion_tokens", parameters.get("max_tokens"))
    if limit is not None and args.mode == "batch":
        parameters.update(max_tokens=limit, max_completion_tokens=limit)
    if args.mode == "streaming":
        if args.provider:
            provider["only"] = [args.provider]
            provider["allow_fallbacks"] = False
        parameters["provider"] = {**provider, "require_parameters": True}
        if args.session_id:
            parameters["session_id"] = args.session_id
    # A shared array schema also works when Google combines requests into one batch schema.
    schema = Classifications.model_json_schema()
    if args.model.startswith("anthropic/"):
        schema = transform_schema(schema)
    response_format = {"type": "json_schema", "json_schema": {
        "name": "codebook_classifications", "strict": True, "schema": schema,
    }}
    payload = {"endpoint": "/v1/chat/completions", "model": args.model}
    if args.mode == "batch" and args.provider:
        payload["provider"] = {"only": [args.provider]}
    payload["requests"] = []
    submitted = []
    for run in runs:
        for start in range(0, len(documents), args.documents_per_request):
            group = documents[start:start + args.documents_per_request]
            custom_id = f"run-{run}-request-{start // args.documents_per_request + 1}"
            payload["requests"].append({"custom_id": custom_id, "body": {
                **parameters, "model": args.model, "stream": args.mode == "streaming", "response_format": response_format,
                "messages": [
                    {"role": "system", "content": prompt + f"\n\nThere are {len(group)} documents in this request."},
                    {"role": "user", "content": json.dumps(group, ensure_ascii=False)},
                ],
            }})
            submitted.append({"custom_id": custom_id, "run": run,
                              "document_ids": [doc["document_id"] for doc in group]})
    meta = {
        "batch_id": None,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "data": data.to_dicts(), "submitted_requests": submitted,
    }
    return payload, meta


def submit_batch(args, api_key):
    payload, meta = prepare_batch(args)
    root = args.out_folder / safe_name(args.nametag) / safe_name(args.model.split(":", 1)[0]) / safe_name(args.codebook_path.stem)
    if args.dry_run:
        folder = root / "dry_run"
    elif args.mode == "streaming":
        folder = root / datetime.now(timezone.utc).strftime("streaming_run_%Y%m%d_%H%M")
    else:
        submission = api_request("POST", api_key, payload=payload)
        meta["batch_id"] = submission["id"]
        meta["submission"] = submission
        folder = root / safe_name(meta["batch_id"])
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "batch_request.json", payload)
    write_json(folder / "batch_meta.json", meta)
    print(f"{len(payload['requests'])} requests; metadata: {folder / 'batch_meta.json'}")
    if args.dry_run:
        print("Dry run: no API calls made.")
    elif args.mode == "streaming":
        run_streaming(payload, meta, folder, api_key)
    else:
        print(f'uv run "{Path(__file__).resolve()}" retrieve --meta-path "{(folder / "batch_meta.json").resolve()}"')


def stream_request(body, api_key):
    content, event_lines = [], []
    result, finish_reason = {}, None
    with requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body, stream=True, timeout=(30, 600),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith(b"data:"):
                event_lines.append(line[5:].strip().decode("utf-8"))
            elif not line and event_lines:
                event = "\n".join(event_lines)
                event_lines = []
                if event == "[DONE]":
                    break
                chunk = json.loads(event)
                if chunk.get("error"):
                    raise ValueError(chunk["error"])
                result.update({key: chunk[key] for key in
                               ("id", "model", "provider", "usage", "created") if key in chunk})
                for choice in chunk.get("choices", []):
                    content.append(choice.get("delta", {}).get("content") or "")
                    finish_reason = choice.get("finish_reason") or finish_reason
        else:
            raise ValueError("Stream ended before [DONE].")
    result["choices"] = [{"finish_reason": finish_reason,
                          "message": {"role": "assistant", "content": "".join(content)}}]
    return result


def run_streaming(payload, meta, folder, api_key):
    results = {"status": "completed", "results": []}
    requests_by_id = {item["custom_id"]: item for item in payload["requests"]}
    finished_requests = []
    for run in dict.fromkeys(item["run"] for item in meta["submitted_requests"]):
        run_requests = [item for item in meta["submitted_requests"] if item["run"] == run]
        for submitted in run_requests:
            request = requests_by_id[submitted["custom_id"]]
            print(f"Streaming {request['custom_id']}", flush=True)
            item = {"custom_id": request["custom_id"]}
            try:
                item["response"] = {"status_code": 200, "body": stream_request(request["body"], api_key)}
            except (requests.RequestException, ValueError) as error:
                item["error"] = str(error)
                print(f"{request['custom_id']}: {error}")
            results["results"].append(item)
            # Checkpoint each response, then export complete runs before sending the next run.
            write_json(folder / "batch_results.json", results)
        finished_requests.extend(run_requests)
        export_results({**meta, "submitted_requests": finished_requests}, results, folder)


def read_classifications(result, expected_ids):
    if result.get("error"):
        raise ValueError(result["error"])
    response = result["response"]
    if response["status_code"] != 200:
        raise ValueError(f"HTTP {response['status_code']}: {response.get('body')}")
    choice = response["body"]["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
    content = choice["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*\n?(.*?)\s*```$", r"\1", content, flags=re.DOTALL)
    rows = Classifications.model_validate_json(content).model_dump()["classifications"]
    if Counter(row["document_id"] for row in rows) != Counter(expected_ids):
        raise ValueError("Missing, duplicate, or unexpected document IDs.")
    return rows


def retrieve_batch(meta_path, api_key):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["arguments"].get("mode") == "streaming":
        batch = json.loads(meta_path.with_name("batch_results.json").read_text(encoding="utf-8"))
    else:
        batch = api_request("GET", api_key, meta["batch_id"])
    write_json(meta_path.with_name("batch_results.json"), batch)
    print(f"Results {meta['batch_id'] or meta_path.parent.name}: {batch['status']}")
    if batch["status"] not in {"completed", "failed", "expired", "cancelled"}:
        print("Still processing. Run the same retrieve command later.")
        return 0
    return export_results(meta, batch, meta_path.parent)


def export_results(meta, batch, output_folder):
    model = safe_name(meta["arguments"]["model"].split(":", 1)[0])
    results = {item["custom_id"]: item for item in batch.get("results") or []}
    data = pl.DataFrame(meta["data"], infer_schema_length=None)
    summary = []
    for run in dict.fromkeys(item["run"] for item in meta["submitted_requests"]):
        rows, errors = [], []
        for item in meta["submitted_requests"]:
            if item["run"] != run:
                continue
            try:
                if batch["status"] != "completed" or batch.get("error"):
                    raise ValueError(f"Batch {batch['status']}: {batch.get('error')}")
                rows.extend(read_classifications(results[item["custom_id"]], item["document_ids"]))
            except (ValueError, TypeError, KeyError, IndexError) as error:
                errors.append({"custom_id": item["custom_id"], "error": str(error)})
        if not errors:
            classified = data.join(pl.DataFrame(rows), on="document_id", how="left",
                                   validate="1:1", maintain_order="left", suffix="_classification")
            folder = output_folder / f"run_{run}"
            folder.mkdir(parents=True, exist_ok=True)
            classified.write_csv(folder / f"classifications_{model}.csv")
        summary.append({"run": run, "status": "error" if errors else "success", "errors": errors})
    write_json(output_folder / "retrieval_summary.json", summary)
    retry = [str(item["run"]) for item in summary if item["status"] != "success"]
    print(f"Saved {len(summary) - len(retry)}/{len(summary)} runs.")
    if retry:
        print("Retry with a new submission: --runs " + " ".join(retry))
    return int(bool(retry))


def main():
    args = parse_args()
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if args.command == "submit":
        submit_batch(args, api_key)
    else:
        raise SystemExit(retrieve_batch(args.meta_path, api_key))


if __name__ == "__main__":
    main()

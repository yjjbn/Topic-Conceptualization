"""Batch N independent classifications of the full document set, without a codebook.

From this folder:
  uv run classify_batch.py submit --model "moonshotai/kimi-k3" --repeats 50 --data-path ./data/jung_codebookdev_200.csv --prompt-path ./prompts/prompt_classify_only.txt --nametag jung --dry-run
  uv run classify_batch.py retrieve --meta-path PATH_PRINTED_BY_SUBMIT

Supply your classification prompt with --prompt-path.
Use --mode streaming for sequential regular requests and automatic export.
Streaming folders are named streaming_run_YYYYMMDD_HHMM (UTC).
Use --documents-per-request to split each run into chunks (otherwise all documents).
Retrieval combines each run's chunks into one CSV. Successful runs are
saved under out/NAMETAG/MODEL/BATCH_ID/run_N. Warnings/errors stay in the batch
summary and raw batch results; they do not get run folders.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import polars as pl
import requests
from dotenv import load_dotenv
from pydantic import Field, create_model
from anthropic import transform_schema
from codebook_formats import StrictModel, ExplainedClassification

from classify_create_codebook import (
    load_parameters, read_data, safe_name, clean_response_json, write_json,
)

BATCHES_URL = "https://openrouter.ai/api/v1/batches"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit", help="classify using batch or streaming requests")
    submit.add_argument("--model", required=True)
    submit.add_argument("--mode", choices=["batch", "streaming"], default="batch")
    submit.add_argument("--provider", help="use only this OpenRouter provider, in either mode")
    submit.add_argument("--session-id", help="session ID for streaming requests; ignored in batch mode")
    submit.add_argument("--repeats", type=int, default=50)
    submit.add_argument("--start-run", "--run", type=int, default=1,
                        help="first run number (default: 1)")
    submit.add_argument("--runs", type=int, nargs="+",
                        help="explicit run numbers, e.g. --runs 5 40; overrides repeats/start-run")
    submit.add_argument("--data-path", type=Path, required=True)
    submit.add_argument("--prompt-path", type=Path, required=True)
    submit.add_argument("--test-rows", type=int)
    submit.add_argument("--documents-per-request", type=int,
                        help="documents per chunk; defaults to the full dataset")
    submit.add_argument("--nametag", default="")
    submit.add_argument("--out-folder", type=Path, default=Path("./out"))
    submit.add_argument("--note", default="")
    submit.add_argument("--dry-run", action="store_true",
                        help="save and inspect the payload without contacting the API")
    retrieve = commands.add_parser("retrieve", help="check status and export completed repeats")
    retrieve.add_argument("--meta-path", type=Path, required=True)
    return parser.parse_args()


class DocumentClassification(ExplainedClassification):
    document_id: str


def response_model(document_ids, as_array):
    if not document_ids or len(set(document_ids)) != len(document_ids):
        raise ValueError("Input document IDs must be nonempty and unique.")

    def id_schema(schema):
        value = schema["additionalProperties"]
        schema.update(properties={key: value for key in document_ids},
                      required=document_ids, additionalProperties=False)

    classifications = (
        (list[DocumentClassification], ...)
        if as_array else
        (dict[str, ExplainedClassification], Field(json_schema_extra=id_schema))
    )
    return create_model("ClassificationsOnly", __base__=StrictModel,
                        classifications=classifications)


def build_request(model, prompt, documents, mode="batch", provider=None, session_id=None):
    parameters = load_parameters(model)
    preferences = parameters.pop("provider", {})
    token_limit = parameters.get("max_completion_tokens", parameters.get("max_tokens"))
    if token_limit is not None and mode == "batch":
        # Send both names with the same configured limit for batch providers.
        parameters.update(max_tokens=token_limit, max_completion_tokens=token_limit)
    if mode == "streaming":
        if provider:
            preferences.update(only=[provider], allow_fallbacks=False)
        parameters["provider"] = {**preferences, "require_parameters": True}
        if session_id:
            parameters["session_id"] = session_id
    ids = [doc["document_id"] for doc in documents]
    schema = response_model(ids, model.lower().startswith("google/")).model_json_schema()
    if model.lower().startswith("anthropic/"):
        schema = transform_schema(schema)
    return {
        **parameters,
        "model": model,
        "stream": mode == "streaming",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(documents)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "classifications_only", "strict": True, "schema": schema,
        }},
    }


def read_classifications(response_body, ids, as_array):
    choice = response_body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
    content = clean_response_json(choice["message"]["content"])
    result = response_model(ids, as_array).model_validate_json(content).model_dump()
    classifications = result["classifications"]
    counts = Counter(item["document_id"] for item in classifications) if as_array else Counter(classifications.keys())
    warnings = []
    for label, values in (
        ("Missing", sorted(set(ids) - counts.keys())),
        ("Unexpected", sorted(counts.keys() - set(ids))),
        ("Duplicate", [key for key, count in counts.items() if count > 1]),
    ):
        if values:
            warnings.append(f"{label} document IDs: {', '.join(values)}")
    if warnings:
        return [], warnings
    if as_array:
        rows = classifications
    else:
        rows = [{"document_id": key, **value} for key, value in classifications.items()]
    return rows, []


def export_repeat(meta, run, responses, folder):
    args = meta["arguments"]
    model = safe_name(args["model"].split(":", 1)[0])
    prefix = model if args["test_rows"] is None else f"test_{args['test_rows']}_{model}"
    data = pl.DataFrame(meta["data"], schema={column: pl.String for column in meta["columns"]})
    rows, warnings = [], []
    for item, body in responses:
        ids = item.get("document_ids", data["document_id"].to_list())
        chunk_rows, chunk_warnings = read_classifications(body, ids, meta["as_array"])
        rows.extend(chunk_rows)
        warnings.extend(f"{item['custom_id']}: {warning}" for warning in chunk_warnings)
    if warnings:
        return warnings
    classified = pl.DataFrame(rows, schema={"document_id": pl.String, "label": pl.String, "explanation": pl.String})
    output = data.join(classified, on="document_id", how="left", maintain_order="left",
                       suffix="_classification", validate="1:1")
    folder.mkdir(parents=True, exist_ok=True)
    output.write_csv(folder / f"classifications_{prefix}.csv")
    write_json(folder / f"raw_response_{prefix}.json", [
        {"custom_id": item["custom_id"], "body": body} for item, body in responses
    ])
    write_json(folder / f"note_{prefix}.txt", {
        "note": args["note"], "batch_id": meta["batch_id"], "run": run,
        "arguments": args, "usage": [body.get("usage") for _, body in responses],
    })
    return []


def prepare_batch(args):
    data = read_data(args.data_path, args.test_rows)
    documents = data.select("document_id", "text").to_dicts()
    as_array = args.model.lower().startswith("google/")
    prompt = args.prompt_path.read_text(encoding="utf-8")
    response_model([doc["document_id"] for doc in documents], as_array)
    chunk_size = args.documents_per_request if args.documents_per_request is not None else len(documents)
    run_numbers = args.runs or list(range(args.start_run, args.repeats+1))
    if chunk_size < 1 or not run_numbers or min(run_numbers) < 1 or len(set(run_numbers)) != len(run_numbers):
        raise ValueError("Chunk size and run numbers must be positive; run numbers must be unique.")
    # The stream-parsed API requires endpoint/model before requests.
    payload = {"endpoint": "/v1/chat/completions", "model": args.model}
    if args.mode == "batch" and args.provider:
        payload["provider"] = {"only": [args.provider]}
    payload["requests"] = []
    runs = []
    for run in run_numbers:
        for start in range(0, len(documents), chunk_size):
            group = documents[start:start + chunk_size]
            instructions = prompt + f"\nThere are {len(group)} documents in this request. Classify every document."
            body = build_request(args.model, instructions, group, args.mode, args.provider, args.session_id)
            custom_id = f"run-{run}-request-{start // chunk_size + 1}"
            payload["requests"].append({"custom_id": custom_id, "body": body})
            runs.append({"custom_id": custom_id, "run": run,
                         "document_ids": [doc["document_id"] for doc in group]})
    meta = {
        "batch_id": None,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "as_array": as_array,
        "data": data.to_dicts(), "columns": data.columns,
        "submitted_requests": runs,
    }
    return payload, meta


def api_request(method, url, api_key, payload=None):
    response = requests.request(
        method, url, headers={"Authorization": f"Bearer {api_key}"},
        json=payload, timeout=(30, 600),
    )
    response.raise_for_status()
    return response.json()


def submit_batch(args, api_key):
    payload, meta = prepare_batch(args)
    root = args.out_folder / safe_name(args.nametag) / safe_name(args.model.split(":", 1)[0])
    if args.dry_run:
        folder = root / "dry_run"
    elif args.mode == "streaming":
        folder = root / datetime.now(timezone.utc).strftime("streaming_run_%Y%m%d_%H%M")
    else:
        submission = api_request("POST", BATCHES_URL, api_key, payload)
        meta["batch_id"] = submission["id"]
        meta["submission"] = submission
        folder = root / safe_name(meta["batch_id"])
        print(f"Submitted batch {meta['batch_id']}")
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "batch_meta.json"
    write_json(folder / "batch_request.json", payload)
    write_json(meta_path, meta)
    print(f"Metadata: {meta_path.resolve()}")
    if args.dry_run:
        print("Dry run: no API calls made.")
    elif args.mode == "streaming":
        run_streaming(payload, meta, folder, api_key)
    else:
        print(f'uv run "{Path(__file__).resolve()}" retrieve --meta-path "{meta_path.resolve()}"')


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


def retrieve_batch(meta_path, api_key):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["arguments"].get("mode") == "streaming":
        batch = json.loads(meta_path.with_name("batch_results.json").read_text(encoding="utf-8"))
    else:
        batch = api_request("GET", f"{BATCHES_URL}/{meta['batch_id']}", api_key)
    write_json(meta_path.with_name("batch_results.json"), batch)
    status = batch["status"]
    print(f"Results {meta['batch_id'] or meta_path.parent.name}: {status}")
    if status not in TERMINAL_STATUSES:
        print("Still processing. Run the same retrieve command later.")
        return 0
    return export_results(meta, batch, meta_path.parent)


def export_results(meta, batch, output_folder):
    status = batch["status"]
    results = {item["custom_id"]: item for item in batch.get("results") or []}
    summary = []
    for run in dict.fromkeys(item["run"] for item in meta["submitted_requests"]):
        entry = {"run": run, "status": "error", "warnings": [], "error": None}
        try:
            if status != "completed" or batch.get("error"):
                raise ValueError(f"Batch {status}: {batch.get('error')}")
            responses = []
            for item in meta["submitted_requests"]:
                if item["run"] != run:
                    continue
                result = results[item["custom_id"]]
                if result.get("error"):
                    raise ValueError(f"{item['custom_id']}: {result['error']}")
                response = result["response"]
                if response["status_code"] != 200:
                    raise ValueError(f"{item['custom_id']}: HTTP {response['status_code']}: {response.get('body')}")
                responses.append((item, response["body"]))
            entry["warnings"] = export_repeat(
                meta, run, responses, output_folder / f"run_{run}",
            )
            entry["status"] = "warning" if entry["warnings"] else "success"
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            entry["error"] = str(exc)
            print(f"Run {run}: {exc}")
        summary.append(entry)

    write_json(output_folder / "retrieval_summary.json", summary)
    retry_runs = [str(item["run"]) for item in summary if item["status"] != "success"]
    print(f"Saved {len(summary) - len(retry_runs)}/{len(summary)} runs.")
    if retry_runs:
        print("Retry these runs: --runs " + " ".join(retry_runs))
    return int(status != "completed" or bool(retry_runs))


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

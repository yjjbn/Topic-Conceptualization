"""Reapply each saved codebook to its original documents using its creator model."""
import argparse
import json
import os
import re
from copy import copy
from collections import Counter
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from apply_codebook_batch import prepare_batch, safe_name, submit_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebooks-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nametag", required=True)
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--test-rows", type=int)
    parser.add_argument("--note", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_jobs(args):
    """Recover exact API model names and original run numbers from 01 outputs."""
    jobs = []
    seen = set()
    for path in sorted(args.codebooks_root.glob("run_*/*/codebook_*.json")):
        suffix = path.stem.removeprefix("codebook_")
        note_path = path.with_name(f"note_{suffix}.txt")
        original_path = path.with_name(f"classifications_{suffix}.csv")
        notes = note_path.read_text(encoding="utf-8")
        request = json.loads(notes.split("Full request body sent to API:\n", 1)[1])
        model = request["model"]
        if args.model and model != args.model:
            continue
        run_match = re.fullmatch(r"run_(\d+)", path.parent.parent.name)
        if not run_match or safe_name(model) != path.parent.name:
            raise ValueError(f"Source model/run does not match its folder: {path}")
        key = (model, int(run_match[1]))
        if key in seen:
            raise ValueError(f"Multiple codebooks for model/run {key}; choose an unambiguous source folder.")
        seen.add(key)
        job = copy(args)
        job.model, job.run = key
        job.codebook_path = path
        job.original_classifications_path = original_path
        job.original_request_documents = json.loads(request["messages"][1]["content"])
        jobs.append(job)
    if not jobs:
        raise ValueError("No matching codebooks found.")
    return jobs


def prepare_reapplication(args):
    prepared = prepare_batch(args)
    document_inputs = [d for request in prepared[1] for d in request["documents"]]
    source = None
    if hasattr(args, "original_classifications_path"):
        original = pl.read_csv(args.original_classifications_path, schema_overrides={"document_id": pl.String})
        if original["document_id"].n_unique() != original.height:
            raise ValueError(f"Duplicate original IDs: {args.original_classifications_path}")
        original_by_id = {row["document_id"]: row for row in original.iter_rows(named=True)}
        submitted_by_id = {row["document_id"]: row["text"] for row in args.original_request_documents}
        input_by_id = {row["document_id"]: row["text"] for row in document_inputs}
        if args.test_rows is None and set(input_by_id) != set(submitted_by_id):
            raise ValueError(f"Dataset IDs differ from the original run: {args.codebook_path}")
        for document in document_inputs:
            document_id = document["document_id"]
            old = original_by_id.get(document_id)
            if (not old or old["text"] != document["text"]
                    or submitted_by_id.get(document_id) != document["text"]):
                raise ValueError(f"Original text/ID mismatch for {document_id}: {args.codebook_path}")
            if old.get("label") is not None and str(old["label"]) not in ("0", "1"):
                raise ValueError(f"Invalid original label for {document_id}: {args.original_classifications_path}")
        missing = sum(original_by_id[d["document_id"]].get("label") is None for d in document_inputs)
        if missing:
            print(f"WARNING: {args.model}, run {args.run}: {missing} original labels are missing; comparisons will be null for those texts.")
        source = {
            "model": args.model, "run": args.run,
            "codebook_path": str(args.codebook_path.resolve()),
            "original_classifications_path": str(args.original_classifications_path.resolve()),
            "classifications": [
                {"document_id": d["document_id"],
                 "original_label": (
                     str(original_by_id[d["document_id"]]["label"])
                     if original_by_id[d["document_id"]].get("label") is not None else None
                 ),
                 "original_explanation": original_by_id[d["document_id"]].get("explanation")}
                for d in document_inputs
            ],
        }
    return prepared, source


def main():
    args = parse_args()
    # Validate every pairing before submitting any paid requests.
    jobs = [(job, *prepare_reapplication(job)) for job in discover_jobs(args)]
    print(f"Total batches: {len(jobs)}")
    for model, count in sorted(Counter(job.model for job, _, _ in jobs).items()):
        print(f"  {model}: {count} codebooks")
    if args.dry_run:
        print("Dry run complete; no API requests submitted.")
        return
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    for job, prepared, source in jobs:
        submit_batch(job, prepared, api_key, source=source)


if __name__ == "__main__":
    main()

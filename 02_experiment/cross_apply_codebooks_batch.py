"""Apply every model to every selected codebook, using apply_batch.

Each pairing gets one batch containing all repeats and document chunks.
Retrieve results with apply_batch.py retrieve --meta-path PATH_PRINTED_BY_SUBMIT.
Use --mode streaming to send regular requests and export automatically instead.
"""
import argparse
import os
from copy import copy
from pathlib import Path

from dotenv import load_dotenv

from apply_batch import submit_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch")
    parser.add_argument("--provider", help="use only this OpenRouter provider for all pairings, in either mode")
    parser.add_argument("--session-id", help="session ID for all streaming requests; ignored in batch mode")
    parser.add_argument("--codebook", action="append", required=True, metavar="PATH",
                        help="repeat for each codebook file")
    parser.add_argument("--models", nargs="+", required=True, help="models to apply to every codebook")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--nametag", default="")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--start-run", "--run", type=int, default=1)
    parser.add_argument("--runs", type=int, nargs="+", help="specific run numbers; overrides repeats/start-run")
    parser.add_argument("--documents-per-request", type=int, default=40)
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_experiment(args):
    codebooks = [Path(path) for path in args.codebook]
    models = args.models
    jobs = []
    for model in models:
        for path in codebooks:
            job = copy(args)
            job.model = model
            job.codebook_path = path
            print(f"Application: {model} | Codebook: {path}")
            jobs.append(job)
    print(f"{len(models)} application models x {len(codebooks)} codebooks = {len(jobs)} {args.mode} submissions")
    return jobs


def main():
    args = parse_args()
    jobs = prepare_experiment(args)
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    for job in jobs:
        submit_batch(job, api_key)


if __name__ == "__main__":
    main()

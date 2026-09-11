import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Literal
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, Field, model_validator
from anthropic import transform_schema

class StrictModel(BaseModel, extra="forbid", strict=True):
    pass


type Label = Literal["0", "1"]


class Classification(StrictModel):
    document_id: str
    label: Label


class Example(StrictModel):
    text: str
    explanation: str


class DecisionRule(StrictModel):
    rule_id: Literal["1", "2", "3", "4", "5"]
    rule: str
    positive_example: Example
    negative_example: Example


class Codebook(StrictModel):
    definition: str
    decision_rules: list[DecisionRule] = Field(min_length=1, max_length=5)


# run like
# uv run classify_create_codebook.py --model "anthropic/claude-sonnet-5" --nametag "elemental" --note "helpful note" --data-path "==PATH==" --prompt-path "==PATH==" --test-rows ==e.g. 5, if you want to test 5 rows==
# will output a .csv with the classifications, and a .json with the codebook

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-id", default=None, help="reuse this ID across runs for OpenRouter sticky routing and prompt cache reuse")
    parser.add_argument("--nametag", default="", help="add a descriptive tag to add to output names, like 'elemental'")
    parser.add_argument("--note", default="", help="add a note if you want")
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev_350.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/prompt_classify-create-codebook_grimmer_elemental.txt"))
    parser.add_argument("--test-rows", type=int, default=None, help="if you want to test a few random rows")
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-output-tokens", type=int, default=50000)
    return parser.parse_args()


def safe_name(value: str) -> str:
    # make name safe for a file name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_response_text(body: dict) -> str:
    return body["choices"][0]["message"]["content"]


def build_response_model(document_ids: list[str]) -> type[BaseModel]:
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    expected_ids = set(document_ids)

    class ClassificationsAndCodebook(StrictModel):
        classifications: list[Classification] = Field(
            # Request the exact count without rejecting partial responses locally.
            json_schema_extra={"minItems": len(document_ids), "maxItems": len(document_ids)},
        )
        codebook: Codebook

        @model_validator(mode="after")
        def validate_document_ids(self):
            returned_ids = [item.document_id for item in self.classifications]
            counts = Counter(returned_ids)
            missing = sorted(expected_ids - counts.keys())
            unexpected = sorted(counts.keys() - expected_ids)
            duplicates = sorted(key for key, count in counts.items() if count > 1)
            if missing or unexpected or duplicates:
                print(
                    f"WARNING: Returned classifications for "
                    f"Submitted {len(expected_ids)} docs, returned {len(counts.keys())}. "
                )
                for description, ids in (
                    ("Missing", missing),
                    ("Unexpected", unexpected),
                    ("Duplicate", duplicates),
                ):
                    if ids:
                        print(f"WARNING: {description} document IDs: {', '.join(ids)}")
            return self

    return ClassificationsAndCodebook


def build_response_format(document_ids: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classifications_and_codebook",
            "strict": True,
            "schema": build_response_model(document_ids).model_json_schema(by_alias=True),
        },
    }


def send_request(api_key: str, request_body: dict) -> dict:
    with OpenRouter(api_key=api_key) as client:
        try:
            response = client.chat.send(
                **request_body, stream=False, timeout_ms=120000
            )
            return response.model_dump(mode="json", by_alias=True)
        except Exception as exc:
            print(getattr(exc, "body", str(exc)))
            raise


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    prompt = args.prompt_path.read_text(encoding="utf-8")
    data = pl.read_csv(args.data_path).with_columns(pl.all().cast(pl.String))

    # if you are sampling rows for a test run
    if args.test_rows is not None:
        data = data.sample(n=args.test_rows, seed=1)

    # format data into a array of json objects, each one is a text
    documents = [
        {"document_id": str(row[args.id_column]), "text": str(row[args.text_column])}
        for row in data.iter_rows(named=True)
    ]
    document_ids = [document["document_id"] for document in documents]
    response_format = build_response_format(document_ids)
    if "anthropic" in args.model:
        response_format["json_schema"]["schema"] = transform_schema(
            response_format["json_schema"]["schema"]
        )
    prompt += (
        "\n\nReturn classifications as an array of objects, each containing "
        'a "document_id" and a "label" ("0" or "1"), '
        "with every input ID included exactly once. "
        f"There are {len(documents)} documents in this request. "
        "Return the codebook alongside classifications as specified in the schema."
    )

    print(
        f"Classifying {len(documents)} documents and generating a codebook "
        f"with {args.model}..."
    )

    response_body = send_request(
        api_key,
        {
            "model": args.model,
            **({"session_id": args.session_id} if args.session_id is not None else {}),
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(documents)},
            ],
            "response_format": response_format,
            "provider": {"require_parameters": True,
                        #  "only": ["anthropic", "openai"],
            },
            "plugins": [
                {"id": "response-healing"}
            ],
            # "max_completion_tokens": args.max_output_tokens,
        },
    )

    # prep file names
    model_name = safe_name(args.model)
    nametag = safe_name(args.nametag)
    output_folder = (
        args.out_folder
        / f"{model_name}"
        / f"{nametag}"
    )
    if args.test_rows is None:
        file_prefix = f"{model_name}_{nametag}"
    else:
        file_prefix = f"test_{args.test_rows}_{model_name}_{nametag}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save the full SDK response before extracting or parsing message content.
    # This includes reasoning fields, usage, IDs, and all choices when returned.
    raw_response_path = output_folder / f"raw_response_{file_prefix}.json"
    raw_response_path.write_text(
        json.dumps(response_body["choices"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_text = get_response_text(response_body)

    classifications_path = output_folder / f"classifications_{file_prefix}.csv"
    codebook_path = output_folder / f"codebook_{file_prefix}.json"

    response_model = build_response_model(document_ids)
    result = response_model.model_validate_json(output_text).model_dump(mode="json", by_alias=True)
                        
    classifications = pl.DataFrame(
        result["classifications"],
        schema={"document_id": pl.String, "label": pl.String},
    )
    # Keep one row per input text, even when classifications are missing.
    # Duplicate predictions have already been reported; retain the first.
    classifications = classifications.unique(subset="document_id", keep="first")
    data.join(
        classifications,
        left_on=args.id_column,
        right_on="document_id",
        how="left",
        maintain_order="left",
        suffix="_classification",
    ).write_csv(classifications_path)

    codebook_path.write_text(
        json.dumps(result["codebook"], indent=2) + "\n",
        encoding="utf-8",
    )

    usage = response_body.get("usage") or {}
    cache_details = usage.get("prompt_tokens_details") or {}
    note_text = args.note
    note_text += "\n\nOriginal command-line arguments:\n" + json.dumps(
        sys.argv, indent=2, ensure_ascii=False
    )
    note_text += "\n\nParsed arguments (including defaults):\n" + json.dumps(
        vars(args), indent=2, ensure_ascii=False, default=str
    )
    if args.session_id is not None:
        note_text += f"\n\nSession ID: {args.session_id}"
    masked_key = ("..." + api_key[-4:]) if api_key else "not available"
    usage_details = {
        "Generation ID": response_body.get("id"),
        "Cost (OpenRouter credits)": usage.get("cost"),
        "Input tokens": usage.get("prompt_tokens"),
        "Output tokens": usage.get("completion_tokens"),
        "Cached input tokens": cache_details.get("cached_tokens"),
        "Cache write tokens": cache_details.get("cache_write_tokens"),
        "API key (masked)": masked_key,
    }
    note_text += "\n\n" + "\n".join(
        f"{name}: {value if value is not None else 'not reported'}"
        for name, value in usage_details.items()
    ) + "\n"
    (output_folder / f"note_{file_prefix}.txt").write_text(note_text, encoding="utf-8")

    print("parsed and saved classifications and codebook")

if __name__ == "__main__":
    main()

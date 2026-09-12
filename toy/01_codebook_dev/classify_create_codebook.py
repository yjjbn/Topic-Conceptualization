import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, Field, create_model, model_validator
from anthropic import transform_schema

from codebook_formats import (
    FORMATS, Label, StrictModel, DecisionRulesCodebook,
)

# run like
# uv run classify_create_codebook.py --model "anthropic/claude-sonnet-5" --nametag "elemental" --note "helpful note" --data-path "==PATH==" --format "explanation+codebookLLM" --test-rows ==e.g. 5, if you want to test 5 rows==
# will output a .csv with the classifications, and a .json with the codebook

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-id", default=None, help="for OpenRouter sticky routing and prompt cache reuse")
    parser.add_argument("--nametag", default="", help="add a descriptive tag to add to output names, also creates a folder")
    parser.add_argument("--run", type=int, default=1, help="positive repetition number (default: 1)")
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev_100.csv"))
    parser.add_argument("--format", choices=FORMATS, default="explanation_codebookLLM")
    parser.add_argument("--test-rows", type=int, default=None, help="if you want to test a few random rows")
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--note", default="", help="add a note if you want")
    # parser.add_argument("--max-output-tokens", type=int, default=50000)
    args = parser.parse_args()
    if args.run < 1:
        parser.error("--run must be at least 1")
    return args


def safe_name(value: str) -> str:
    # make name safe for a file name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def get_response_text(body: dict) -> str:
    return body["choices"][0]["message"]["content"]


def build_response_model(
    document_ids: list[str],
    codebook_model: type[StrictModel] = DecisionRulesCodebook,
    classification_type=Label,
    *, as_array: bool = False,
) -> type[BaseModel]:
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    expected_ids = set(document_ids)

    def require_document_ids(schema):
        # Reuse Pydantic's generated value schema, including any $ref.
        value_schema = schema["additionalProperties"]
        schema.update(
            properties={document_id: value_schema for document_id in document_ids},
            required=list(document_ids),
            additionalProperties=False,
        )

    class ClassificationsResponse(StrictModel):
        @model_validator(mode="after")
        def validate_document_ids(self):
            counts = Counter(
                item.document_id for item in self.classifications
            ) if as_array else Counter(self.classifications.keys())
            returned_ids = counts.keys()
            duplicates = [key for key, count in counts.items() if count > 1]
            missing = sorted(expected_ids - returned_ids)
            unexpected = sorted(returned_ids - expected_ids)
            if missing or unexpected or duplicates:
                print(
                    f"WARNING: Returned classifications for "
                    f"Submitted {len(expected_ids)} docs, returned {len(returned_ids)}. "
                )
                for description, ids in (
                    ("Missing", missing),
                    ("Unexpected", unexpected),
                    ("Duplicate", duplicates),
                ):
                    if ids:
                        print(f"WARNING: {description} document IDs: {', '.join(ids)}")
            return self

    if as_array:
        if isinstance(classification_type, type) and issubclass(classification_type, BaseModel):
            item_model = create_model(
                "DocumentClassification", __base__=classification_type,
                document_id=(str, ...),
            )
        else:
            item_model = create_model(
                "DocumentClassification", __base__=StrictModel,
                document_id=(str, ...), label=(classification_type, ...),
            )
        classifications_field = (list[item_model], ...)
    else:
        classifications_field = (
            dict[str, classification_type], Field(json_schema_extra=require_document_ids)
        )

    return create_model(
        "ClassificationsAndCodebook",
        __base__=ClassificationsResponse,
        classifications=classifications_field,
        codebook=(codebook_model, ...),
    )


def build_response_format(
    document_ids: list[str],
    codebook_model: type[StrictModel] = DecisionRulesCodebook,
    classification_type=Label,
    *, as_array: bool = False,
) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classifications_and_codebook",
            "strict": True,
            "schema": build_response_model(document_ids, codebook_model, classification_type, as_array=as_array).model_json_schema(by_alias=True),
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
    config = FORMATS[args.format]
    codebook_model = config.codebook_model
    classification_type = config.classification_type
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    prompt_path = Path(__file__).resolve().parent / "prompts" / config.prompt
    prompt = prompt_path.read_text(encoding="utf-8")
    data = pl.read_csv(args.data_path).with_columns(pl.all().cast(pl.String))

    # if you are sampling rows for a test run
    if args.test_rows is not None:
        data = data.sample(n=args.test_rows, seed=1)

    # format data into a array of json objects, each one is a text
    documents = [
        {"document_id": str(row["document_id"]), "text": str(row["text"])}
        for row in data.iter_rows(named=True)
    ]
    document_ids = [document["document_id"] for document in documents]
    as_array = args.model.lower().startswith("google/")
    response_format = build_response_format(document_ids, codebook_model, classification_type, as_array=as_array)
    if "anthropic" in args.model:
        response_format["json_schema"]["schema"] = transform_schema(
            response_format["json_schema"]["schema"]
        )

    prompt += (
        f"There are {len(documents)} documents in this request. "
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
        / f"{nametag}"
        / f"run_{args.run}"
        / f"{model_name}"
    )
    if args.test_rows is None:
        file_prefix = f"{nametag}_run_{args.run}_{model_name}"
    else:
        file_prefix = f"test_{args.test_rows}_{nametag}_run_{args.run}_{model_name}"
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

    response_model = build_response_model(document_ids, codebook_model, classification_type, as_array=as_array)
    result = response_model.model_validate_json(output_text).model_dump(mode="json", by_alias=True)
    if as_array:
        # Normalize Google output for the existing CSV export; retain the first duplicate.
        classifications_by_id = {}
        for item in result["classifications"]:
            document_id = item["document_id"]
            value = {key: value for key, value in item.items() if key != "document_id"}
            classifications_by_id.setdefault(document_id, value)
        result["classifications"] = classifications_by_id

                        
    classifications = pl.DataFrame(
        [
            {"document_id": document_id, **(
                value if isinstance(value, dict) else {"label": value}
            )}
            for document_id, value in result["classifications"].items()
        ],
        schema={
            "document_id": pl.String,
            **{field: pl.String for field in (
                classification_type.model_fields
                if isinstance(classification_type, type) and issubclass(classification_type, BaseModel)
                else ["label"]
            )},
        },
    )
    # Keep one row per input text, even when classifications are missing.
    data.join(
        classifications,
        on="document_id",
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
    note_text += "\n\nSelected format:\n" + json.dumps(
        {
            "format": args.format,
            "prompt_path": str(prompt_path.resolve()),
            "codebook_model": codebook_model.__name__,
            "classification_type": (
                classification_type.__name__
                if isinstance(classification_type, type) else str(classification_type)
            ),
            "document_count": len(document_ids),
        },
        indent=2, ensure_ascii=False,
    )
    note_text += "\n\nResponse format sent to API:\n" + json.dumps(
        response_format, indent=2, ensure_ascii=False
    )
    note_text += "\n\nSystem instructions sent to API:\n" + prompt + "\n"
    (output_folder / f"note_{file_prefix}.txt").write_text(note_text, encoding="utf-8")

    print("parsed and saved classifications and codebook")

if __name__ == "__main__":
    main()

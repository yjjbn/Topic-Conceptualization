import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, Field, create_model, model_validator
from anthropic import transform_schema

from codebook_formats import (
    StrictModel, CodebookLLMCodebook, ExplainedClassification,
)

DEFAULT_CONFIG = Path(__file__).with_name("model_parameters.toml")


def load_parameters(model):
    """Read this model's API parameters from the TOML file."""
    with DEFAULT_CONFIG.open("rb") as file:
        return tomllib.load(file)[model.split(":", 1)[0]]

# run like
# uv run classify_create_codebook.py --model "anthropic/claude-sonnet-5" --nametag "elemental" --note "helpful note" --data-path "==PATH==" --prompt-path "./prompts/prompt_explain_codebookLLM_grimmer.txt" --test-rows ==e.g. 5, if you want to test 5 rows==
# will output a .csv with the classifications, and a .json with the codebook

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-id", default=None, help="for OpenRouter sticky routing and prompt cache reuse")
    parser.add_argument("--nametag", default="", help="add a descriptive tag to add to output names, also creates a folder")
    parser.add_argument("--run", type=int, default=1, help="run count")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--prompt-path", type=Path)
    parser.add_argument("--test-rows", type=int, default=None, help="if you want to test a few random rows")
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--note", default="", help="add a note if you want")
    return parser.parse_args()


def safe_name(value: str) -> str:
    # make name safe for a file name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def clean_response_json(text: str) -> str:
    """Remove JSON presentation wrappers without repairing the JSON itself."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return re.sub(r"^json\s*(?=[{\[])", "", text, count=1, flags=re.IGNORECASE)


def resolve_example_texts(
    codebook: dict, documents: list[dict], warning_messages: list[str],
) -> None:
    """Replace example document IDs with verbatim source text for export."""
    texts_by_id = {document["document_id"]: document["text"] for document in documents}
    example_groups = [
        (field, codebook.get(field, []))
        for field in ("positive_examples", "negative_examples")
    ]
    for field, examples in example_groups:
        for example in examples:
            if "document_id" not in example:
                continue
            document_id = example["document_id"]
            if document_id in texts_by_id:
                example["text"] = texts_by_id[document_id]
            else:
                example["text"] = None
                message = f"WARNING: Unknown document ID in codebook {field}: {document_id}"
                print(message)
                warning_messages.append(message)
            explanation = example["explanation"]
            text = example["text"]
            example.clear()
            example.update(text=text, explanation=explanation)


def build_response_model(
    document_ids: list[str],
    *, as_array: bool = False,
    warning_messages: list[str] | None = None,
) -> type[BaseModel]:
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    if not document_ids:
        raise ValueError("At least one document is required.")
    expected_ids = set(document_ids)

    def report_warning(message: str) -> None:
        print(message)
        if warning_messages is not None:
            warning_messages.append(message)

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
                report_warning(
                    f"WARNING: Returned classifications for "
                    f"Submitted {len(expected_ids)} docs, returned {len(returned_ids)}. "
                )
                for description, ids in (
                    ("Missing", missing),
                    ("Unexpected", unexpected),
                    ("Duplicate", duplicates),
                ):
                    if ids:
                        report_warning(f"WARNING: {description} document IDs: {', '.join(ids)}")
            return self

    # for google models, need to specify slightly diff structure
    if as_array:
        item_model = create_model(
            "DocumentClassification", __base__=ExplainedClassification,
            document_id=(str, ...),
        )
        classifications_field = (
            list[item_model],
            Field(min_length=len(document_ids), max_length=len(document_ids)),
        )
    else:
        classifications_field = (
            dict[str, ExplainedClassification], Field(json_schema_extra=require_document_ids)
        )

    return create_model(
        "ClassificationsAndCodebook",
        __base__=ClassificationsResponse,
        classifications=classifications_field,
        codebook=(CodebookLLMCodebook, ...),
    )


def build_response_format(
    document_ids: list[str],
    *, as_array: bool = False,
) -> dict:
    schema = build_response_model(
        document_ids, as_array=as_array,
    ).model_json_schema(by_alias=True)

    def constrain_document_ids(node):
        """Constrain classification and example IDs, including nested definitions."""
        if isinstance(node, dict):
            properties = node.get("properties", {})
            if "document_id" in properties:
                properties["document_id"].update(
                    description="Copy the exact document_id from the input. Never generate or renumber IDs.",
                )
                # Google can reject large enums in structured-output schemas.
                # Keep its array schema small; check returned IDs locally.
                if not as_array:
                    properties["document_id"]["enum"] = list(document_ids)
            for value in node.values():
                constrain_document_ids(value)
        elif isinstance(node, list):
            for value in node:
                constrain_document_ids(value)

    constrain_document_ids(schema)
    if as_array:
        # Keep exact-length validation locally, but avoid sending large array
        # bounds that can cause Google to reject the structured-output schema.
        classifications_schema = schema["properties"]["classifications"]
        classifications_schema.pop("minItems", None)
        classifications_schema.pop("maxItems", None)
        classifications_schema["description"] = (
            f"Return exactly {len(document_ids)} classifications, one per input document. "
            "Copy each original document_id exactly, with no omissions or duplicates."
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classifications_and_codebook",
            "strict": True,
            "schema": schema,
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_data(path, test_rows=None):
    data = pl.read_csv(path, schema_overrides={"document_id": pl.String}).with_columns(
        pl.all().cast(pl.String)
    )
    if test_rows is not None:
        data = data.sample(n=test_rows, seed=1)
    documents = data.select("document_id", "text")
    if any(documents.null_count().row(0)):
        raise ValueError("Document IDs and texts cannot be null.")
    return data


def build_request(model, prompt, documents):
    response_format = build_response_format(
        [document["document_id"] for document in documents],
        as_array=model.lower().startswith("google/"),
    )
    if model.lower().startswith("anthropic/"):
        response_format["json_schema"]["schema"] = transform_schema(
            response_format["json_schema"]["schema"]
        )
    return {
        **load_parameters(model),
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(documents)},
        ],
        "response_format": response_format,
    }


def save_results(data, response_body, output_folder, file_prefix, as_array, skip_warnings=False):
    """Validate and export one response; keep warnings for the run notes."""
    documents = data.select("document_id", "text").to_dicts()
    document_ids = [document["document_id"] for document in documents]
    classifications_path = output_folder / f"classifications_{file_prefix}.csv"
    codebook_path = output_folder / f"codebook_{file_prefix}.json"
    warning_messages: list[str] = []
    output_text = response_body["choices"][0]["message"]["content"]
    cleaned_output_text = clean_response_json(output_text)
    response_model = build_response_model(
        document_ids,
        as_array=as_array, warning_messages=warning_messages,
    )
    result = response_model.model_validate_json(cleaned_output_text).model_dump(mode="json", by_alias=True)
    if as_array:
        # Normalize Google output for the existing CSV export; retain the first duplicate.
        classifications_by_id = {}
        for item in result["classifications"]:
            document_id = item["document_id"]
            value = {key: value for key, value in item.items() if key != "document_id"}
            classifications_by_id.setdefault(document_id, value)
        result["classifications"] = classifications_by_id

    resolve_example_texts(result["codebook"], documents, warning_messages)
    if skip_warnings and warning_messages:
        return warning_messages

    classifications = pl.DataFrame(
        [
            {"document_id": document_id, **value}
            for document_id, value in result["classifications"].items()
        ],
        schema={
            "document_id": pl.String,
            **{field: pl.String for field in ExplainedClassification.model_fields},
        },
    )
    # Keep one row per input text, even when classifications are missing.
    output = data.join(
        classifications,
        on="document_id",
        how="left",
        maintain_order="left",
        suffix="_classification",
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    output.write_csv(classifications_path)
    write_json(codebook_path, result["codebook"])

    return warning_messages


def send_request(api_key: str, request_body: dict) -> dict:
    with OpenRouter(api_key=api_key) as client:
        try:
            response = client.chat.send(
                **request_body, timeout_ms=120000
            )
            return response.model_dump(mode="json", by_alias=True)
        except Exception as exc:
            print(getattr(exc, "body", str(exc)))
            raise


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment or .env file.")
    data = read_data(args.data_path, args.test_rows)
    documents = data.select("document_id", "text").to_dicts()
    prompt = args.prompt_path.read_text(encoding="utf-8")
    prompt += f"\n\nThere are {len(documents)} documents in this request. "
    request_body = build_request(args.model, prompt, documents)
    request_body["provider"] = {**request_body.get("provider", {}), "require_parameters": True}
    if args.session_id is not None:
        request_body["session_id"] = args.session_id
    as_array = args.model.lower().startswith("google/")
    print(f"Classifying {len(documents)} documents and generating a codebook with {args.model}...")
    response_body = send_request(api_key, request_body)

    # prep file names
    model_name = safe_name(args.model.split(":", 1)[0])
    output_folder = args.out_folder / safe_name(args.nametag) / f"run_{args.run}" / model_name
    file_prefix = model_name if args.test_rows is None else f"test_{args.test_rows}_{model_name}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Preserve the raw choices before validating or exporting model content.
    write_json(output_folder / f"raw_response_{file_prefix}.json", response_body["choices"])
    warning_messages = save_results(data, response_body, output_folder, file_prefix, as_array)

    usage = response_body.get("usage") or {}
    note_text = args.note
    note_text += "\n\nOriginal command-line arguments:\n" + json.dumps(
        sys.argv, indent=2, ensure_ascii=False
    )

    usage_details = {
        "Generation ID": response_body.get("id"),
        "Cost (OpenRouter credits)": usage.get("cost"),
        "Input tokens": usage.get("prompt_tokens"),
        "Output tokens": usage.get("completion_tokens"),
        "API key (masked)": ("..." + api_key[-4:]),
    }
    note_text += "\n\nUsage details:\n" + "\n".join(
        f"{name}: {value if value is not None else 'not reported'}"
        for name, value in usage_details.items()
    ) + "\n"
    if warning_messages:
        note_text += "\n\nWarnings:\n" + "\n".join(warning_messages) + "\n"
    note_text += "\n\nFull request body sent to API:\n" + json.dumps(
        request_body, indent=2, ensure_ascii=False
    ) + "\n"
    (output_folder / f"note_{file_prefix}.txt").write_text(note_text, encoding="utf-8")

    print("parsed and saved classifications and codebook")

if __name__ == "__main__":
    main()

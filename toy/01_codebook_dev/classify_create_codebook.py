import argparse
import json
import os
import re
from typing import Literal
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, Field, create_model, model_validator
from anthropic import transform_schema

class StrictModel(BaseModel, extra="forbid", strict=True):
    pass


type Label = Literal["0", "1"]


class Example(StrictModel):
    text: str
    explanation: str


class DecisionRulesCodebook(StrictModel):

    class DecisionRule(StrictModel):
        rule_id: Literal["1", "2", "3", "4", "5"]
        rule: str
        positive_example: Example
        negative_example: Example

    definition: str
    decision_rules: list[DecisionRule] = Field(min_length=1, max_length=5)

class CodebookLLMCodebook(StrictModel):

    definition: str
    clarification: str
    positive_examples: list[Example] = Field(min_length=1, max_length=3)
    negative_clarification: str
    negative_examples: list[Example] = Field(min_length=1, max_length=3)

CODEBOOK_MODELS = {
    "decision_rules": DecisionRulesCodebook,
    "codebook_LLM": CodebookLLMCodebook,
}
# run like
# uv run classify_create_codebook.py --model "anthropic/claude-sonnet-5" --nametag "elemental" --note "helpful note" --data-path "==PATH==" --prompt-path "==PATH==" --test-rows ==e.g. 5, if you want to test 5 rows==
# will output a .csv with the classifications, and a .json with the codebook

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-id", default=None, help="reuse this ID across runs for OpenRouter sticky routing and prompt cache reuse")
    parser.add_argument("--nametag", default="", help="add a descriptive tag to add to output names, like 'elemental'")
    parser.add_argument("--note", default="", help="add a note if you want")
    parser.add_argument("--data-path", type=Path, default=Path("./data/grimmer_codebookdev_100.csv"))
    parser.add_argument("--prompt-path", type=Path, default=Path("./prompts/prompt_codebookLLM_grimmer.txt"))
    parser.add_argument(
        "--codebook", choices=CODEBOOK_MODELS, default="codebook_LLM",
        help="codebook output format; use a matching --prompt-path",
    )
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


def build_response_model(
    document_ids: list[str],
    codebook_model: type[StrictModel] = DecisionRulesCodebook,
) -> type[BaseModel]:
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique.")
    expected_ids = set(document_ids)

    class ClassificationsResponse(StrictModel):
        classifications: dict[str, Label] = Field(
            # Require every ID in the request schema, but allow partial results locally.
            json_schema_extra={
                "properties": {
                    document_id: {"type": "string", "enum": ["0", "1"]}
                    for document_id in document_ids
                },
                "required": list(document_ids),
                "additionalProperties": False,
            },
        )

        @model_validator(mode="after")
        def validate_document_ids(self):
            returned_ids = self.classifications.keys()
            missing = sorted(expected_ids - returned_ids)
            unexpected = sorted(returned_ids - expected_ids)
            if missing or unexpected:
                print(
                    f"WARNING: Returned classifications for "
                    f"Submitted {len(expected_ids)} docs, returned {len(returned_ids)}. "
                )
                for description, ids in (
                    ("Missing", missing),
                    ("Unexpected", unexpected),
                ):
                    if ids:
                        print(f"WARNING: {description} document IDs: {', '.join(ids)}")
            return self

    return create_model(
        "ClassificationsAndCodebook",
        __base__=ClassificationsResponse,
        codebook=(codebook_model, ...),
    )


def build_response_format(
    document_ids: list[str],
    codebook_model: type[StrictModel] = DecisionRulesCodebook,
) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classifications_and_codebook",
            "strict": True,
            "schema": build_response_model(document_ids, codebook_model).model_json_schema(by_alias=True),
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
    codebook_model = CODEBOOK_MODELS[args.codebook]
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
    response_format = build_response_format(document_ids, codebook_model)
    if "anthropic" in args.model:
        response_format["json_schema"]["schema"] = transform_schema(
            response_format["json_schema"]["schema"]
        )
    prompt += (
        "\n\nReturn classifications as a JSON object mapping each input document_id to its label (0 or 1), and the codebook as specified in the JSON schema. "
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
        / f"{model_name}"
    )
    if args.test_rows is None:
        file_prefix = f"{nametag}_{model_name}"
    else:
        file_prefix = f"test_{args.test_rows}_{nametag}_{model_name}"
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

    response_model = build_response_model(document_ids, codebook_model)
    result = response_model.model_validate_json(output_text).model_dump(mode="json", by_alias=True)
                        
    classifications = pl.DataFrame(
        [
            {"document_id": document_id, "label": label}
            for document_id, label in result["classifications"].items()
        ],
        schema={"document_id": pl.String, "label": pl.String},
    )
    # Keep one row per input text, even when classifications are missing.
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
    note_text += "\n\nSystem instructions:\n" + prompt + "\n"
    (output_folder / f"note_{file_prefix}.txt").write_text(note_text, encoding="utf-8")

    print("parsed and saved classifications and codebook")

if __name__ == "__main__":
    main()

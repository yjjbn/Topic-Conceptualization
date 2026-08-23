import argparse, json, os, re, time
from pathlib import Path

import requests
from dotenv import load_dotenv

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--classifications-path", type=Path, required=True)
    parser.add_argument("--classification-prompt-path", type=Path, default=Path("./prompts/initial_classify_prompt_grimmer.txt"))
    parser.add_argument("--codebook-prompt-path", type=Path, default=Path("./prompts/write_codebook_grimmer.txt"))
    parser.add_argument("--out-folder", type=Path, default=Path("./out"))
    parser.add_argument("--max-output-tokens", type=int, default=100000)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def clean_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in response:\n{text}")
    return text[start:end + 1]


def send_request(api_key, body, max_retries):
    for attempt in range(max_retries + 1):
        response = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=600,
        )
        if response.status_code not in RETRYABLE_STATUS_CODES or attempt == max_retries:
            break
        try:
            delay = float(response.headers.get("Retry-After"))
        except (TypeError, ValueError):
            delay = 30
        print(f"Retrying in {delay} seconds ({attempt + 1}/{max_retries})")
        time.sleep(delay)

    if not response.ok:
        raise RuntimeError(f"OpenRouter returned {response.status_code}: {response.text}")

    body = response.json()
    if body.get("error"):
        raise RuntimeError(f"OpenRouter response error: {body['error']}")
    return body


def main():
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    classification_prompt = args.classification_prompt_path.read_text(encoding="utf-8")
    codebook_prompt = args.codebook_prompt_path.read_text(encoding="utf-8")
    classifications = json.loads(args.classifications_path.read_text(encoding="utf-8"))

    print(f"Loaded {len(classifications)} classifications.")

    codebook_input = {
        "original_classification_instructions": classification_prompt,
        "classification_records": [
            {
                "document_id": r["document_id"],
                "text": r["text"],
                "label": r["label"],
                "decision_basis": r["decision_basis"],
                "confidence": r.get("confidence"),
            }
            for r in classifications
        ],
    }

    print(f"Generating codebook with {args.model}...")

    response = send_request(
        api_key,
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": codebook_prompt},
                {"role": "user", "content": json.dumps(codebook_input, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_completion_tokens": args.max_output_tokens,
            "reasoning": {"effort": "medium", "exclude": True},
            "response_format": {"type": "json_object"},
        },
        args.max_retries,
    )

    output_text = response["choices"][0]["message"]["content"]
    codebook = json.loads(clean_json(output_text))

    args.out_folder.mkdir(parents=True, exist_ok=True)
    codebook_path = args.out_folder / f"{safe_name(args.model)}_codebook.json"
    codebook_path.write_text(json.dumps(codebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Codebook:", codebook_path)

if __name__ == "__main__":
    main()
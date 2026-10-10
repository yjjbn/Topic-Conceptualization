"""Older notebook helpers. The current workflow is in simple_experiment.py."""

import json
from pathlib import Path

from simple_experiment import (
    load_documents, codebook_input, run_classifications,
    retrieve_classification_batch, parse_classification_runs,
    generate_codebook, revise_codebook, show_agreement, compare_models,
)


def model_folder(model):
    """Use the base model name in paths, excluding API routing slugs."""
    return model.split(":", 1)[0].replace("/", "_")


def parse_codebook(out_dir, codebook_id="A.0.0"):
    """Save the codebook, display its rules, and return its file path."""
    out_dir = Path(out_dir)
    if (out_dir / "batch.json").exists():
        batch = json.loads((out_dir / "batch.json").read_text(encoding="utf-8"))
        response = batch["results"][0]["response"]["body"]
    else:
        response = json.loads((out_dir / "codebook_response.json").read_text(encoding="utf-8"))
    codebook = json.loads(response["choices"][0]["message"]["content"])
    if "codebook" in codebook and "documentation" in codebook:
        documentation = codebook["documentation"]
        codebook = codebook["codebook"]
        (out_dir / f"documentation_{codebook_id}.json").write_text(
            json.dumps(documentation, indent=2, ensure_ascii=False), encoding="utf-8",
        )
    codebook_path = out_dir / f"codebook_{codebook_id}.json"
    codebook_path.write_text(json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(codebook["definition"])
    for rule in codebook["decision_rules"]:
        print(f"{rule['number']}. {rule['rule']}")
    return codebook_path


def review_coding(out_dir, document_order=None):
    """Save classifications, print alpha, show agreement, and return the plotted IDs."""
    results = parse_classification_runs(Path(out_dir))
    return show_agreement(results, document_order=document_order)


def freeze_codebook(codebook_path, out_dir, frozen_id, note):
    """Record the user's freeze decision and copy the accepted codebook."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codebook = json.loads(Path(codebook_path).read_text(encoding="utf-8"))
    (out_dir / f"codebook_{frozen_id}.json").write_text(json.dumps(codebook, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps({"source": str(codebook_path), "frozen_id": frozen_id, "note": note}, indent=2), encoding="utf-8")


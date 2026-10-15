"""Codebook workflow. Start with Experiment; helper functions handle API calls and plots."""

import json
import numpy as np
from pathlib import Path

import polars as pl
import krippendorff
import matplotlib.pyplot as plt
import requests
from openrouter import OpenRouter


# Notebook workflow: the methods below are the steps you call.
class Experiment:
    """One session folder containing data, prompts, codebooks, and results."""
    def __init__(self, session_dir, model_a, model_b, api_key, mode="stream",
                 n_runs=25, provider=None):
        self.session_dir = Path(session_dir).resolve()
        self.root = self.session_dir / "out"
        self.models = {"A": model_a, "B": model_b}
        self.api_key = api_key
        self.mode = mode
        self.n_runs = n_runs
        self.provider = provider

    def _datasets(self, datasets):
        values = [datasets] if isinstance(datasets, int) else list(datasets)
        return sorted(set(values))

    def _documents(self, datasets):
        tables = []
        for number in self._datasets(datasets):
            matches = sorted((self.session_dir / "data").glob(f"*_{number}.csv"))
            if not matches:
                raise FileNotFoundError(f"No file ending in _{number}.csv in {self.session_dir / 'data'}")
            if len(matches) > 1:
                raise ValueError(f"More than one file ends in _{number}.csv: {[path.name for path in matches]}")
            tables.append(load_documents(matches[0]))
        documents = pl.concat(tables)
        if documents["document_id"].n_unique() != documents.height:
            raise ValueError("Document IDs must be unique across the selected datasets.")
        return documents

    def _version_dir(self, version, suffix=None):
        name = f"version-{version:03d}"
        folder = self.root / "codebooks"
        matches = [path for path in folder.glob(f"{name}*")
                   if path.is_dir() and (path.name == name or path.name.startswith(name + "-"))]
        if len(matches) > 1:
            raise ValueError(f"Multiple codebook folders use version {version}.")
        if suffix:
            if any(character in suffix for character in '\\/:*?"<>|'):
                raise ValueError("Use a suffix without special filename characters.")
            target = folder / f"{name}-{suffix}"
            if matches and matches[0] != target:
                raise ValueError(f"Version {version} already exists as {matches[0].name}. Use a new version number.")
            return target
        return matches[0] if matches else folder / name

    def codebook_path(self, version):
        return self._version_dir(version) / "codebook.json"

    def _book(self, version):
        return json.loads(self.codebook_path(version).read_text(encoding="utf-8"))

    def _coding_dir(self, datasets, version, model):
        name = "datasets-" + "-".join(map(str, self._datasets(datasets)))
        book = "no-codebook" if version is None else self._version_dir(version).name
        return self.root / "coding" / name / book / f"model-{model}"

    def _prompt(self, name):
        return (self.session_dir / "prompts" / name).read_text(encoding="utf-8")

    def _record(self, folder, settings):
        """Prevent a reused task from silently changing its inputs."""
        path = folder / "task.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != settings:
                raise ValueError("This task already has different inputs. Use a new version or session_dir.")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    def _ready(self, folder):
        """Check a batch once; never submit from a review cell."""
        batch_path = folder / "batch.json"
        if batch_path.exists():
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            if batch.get("status") != "completed":
                batch = retrieve_classification_batch(self.api_key, folder)
            if batch.get("status") != "completed":
                print(f"Batch: {batch.get('status', 'unknown')}. Rerun this review cell later.")
                return False
        return True

    def _results(self, folder):
        if not self._ready(folder):
            return None
        if not (folder / "batch.json").exists() and not list(folder.glob("run_*.json")):
            raise ValueError("No responses yet. Submit the coding task with preview=False first.")
        results = parse_classification_runs(folder)
        documents = load_documents(folder / "documents.csv")
        expected = set(documents["document_id"].to_list())
        expected_runs = self.n_runs - len(_excluded_runs(folder))
        if not (folder / "batch.json").exists() and results["run"].n_unique() != expected_runs:
            raise ValueError(f"Expected {expected_runs} complete runs; resume the coding cell first.")
        for part in results.partition_by("run"):
            if part.height != len(expected) or set(part["document_id"].to_list()) != expected:
                raise ValueError("Each run must contain all input document IDs exactly once.")
        return results

    def code(self, datasets, version=None, model="A", preview=True, mode=None, note=None):
        """Preview or submit coding; existing streaming runs and batches are reused."""
        folder = self._coding_dir(datasets, version, model)
        if mode is None:
            task_path = folder / "task.json"
            mode = json.loads(task_path.read_text(encoding="utf-8"))["mode"] if task_path.exists() else self.mode
        if mode not in {"stream", "batch"}:
            raise ValueError("mode must be 'stream' or 'batch'.")
        documents = self._documents(datasets)
        book = None if version is None else self._book(version)
        prompt = self._prompt("initial_classification.txt" if book is None else "classify_w_codebook.txt")
        self._record(folder, {
            "kind": "coding", "datasets": self._datasets(datasets), "version": version,
            "model": model, "model_id": self.models[model], "mode": mode,
            "n_runs": self.n_runs, "provider": self.provider,
            "documents": documents.to_dicts(), "prompt": prompt, "codebook": book,
        })
        if note is not None:
            (folder / "note.txt").write_text(note, encoding="utf-8")
        run_classifications(documents, prompt, self.models[model], self.api_key,
                                   folder, mode=mode, n_runs=self.n_runs,
                                   dry_run=preview, codebook=book, provider=self.provider)
        print(f"{'Preview ready' if preview else 'Submitted/resumed'}: model {model}, datasets {self._datasets(datasets)}, "
              f"{'no codebook' if version is None else f'version {version}'}, {self.n_runs} runs ({mode}).",
              folder)

    def review(self, datasets, version=None, model="A"):
        """Show alpha and agreement; persist document order for later reviews."""
        results = self._results(self._coding_dir(datasets, version, model))
        if results is None:
            return
        name = "datasets-" + "-".join(map(str, self._datasets(datasets)))
        path = self.root / "orders" / f"{name}.json"
        order = json.loads(path.read_text()) if path.exists() else None
        order = show_agreement(results, document_order=order)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(order), encoding="utf-8")

    def _summary(self, datasets, version, model):
        folder = self._coding_dir(datasets, version, model)
        numbers = self._datasets(datasets)
        if len(numbers) == 1 or (folder / "batch.json").exists() or list(folder.glob("run_*.json")):
            folders = [folder]
        else:
            folders = [self._coding_dir(number, version, model) for number in numbers]
        summaries = []
        for saved_folder in folders:
            results = self._results(saved_folder)
            if results is None:
                raise ValueError("Coding batch is pending. Review it again when completed.")
            summaries.append(codebook_input(load_documents(saved_folder / "documents.csv"), results))
        summary = pl.concat(summaries)
        if summary["document_id"].n_unique() != summary.height:
            raise ValueError("Document IDs must be unique across the selected datasets.")
        return summary

    def generate_codebook_0(self, datasets=0, version=0, preview=True):
        """Generate the first codebook from model A's coding without a codebook."""
        return self._request_book(datasets, version, None, preview)

    def revise(self, datasets, source_version, version, preview=True,
               suffix=None, note=None, crossmodel=False, prompt_file=None):
        """Create a new numbered version; use model A's coding summaries."""
        if version <= source_version:
            raise ValueError("The new version must be greater than source_version.")
        return self._request_book(datasets, version, source_version, preview, suffix, note, crossmodel, prompt_file)

    def _request_book(self, datasets, version, source_version, preview, suffix=None, note=None,
                      crossmodel=False, prompt_file=None):
        folder = self._version_dir(version, suffix)
        summary = self._summary(datasets, source_version, "A")
        if crossmodel:
            other = self._summary(datasets, source_version, "B")
            if not summary.select("document_id", "text").sort("document_id").equals(
                    other.select("document_id", "text").sort("document_id")):
                raise ValueError("Models A and B must have coded the same documents and text.")
            fields = ["percent_1", "explanation_1", "explanation_0"]
            summary = summary.rename({field: field + "_A" for field in fields})
            other = other.drop("text").rename({field: field + "_B" for field in fields})
            summary = summary.join(other, on="document_id").select(
                "document_id", "text", "percent_1_A", "explanation_1_A", "explanation_0_A",
                "percent_1_B", "explanation_1_B", "explanation_0_B",
            )
        book = None if source_version is None else self._book(source_version)
        if prompt_file is None:
            prompt_file = ("improve_codebook crossmodel.txt" if crossmodel else
                           "generate_A00.txt" if book is None else "improve_codebook.txt")
        prompt = self._prompt(prompt_file)
        self._record(folder, {
            "kind": "codebook", "datasets": self._datasets(datasets), "version": version,
            "source_version": source_version, "model_id": self.models["A"],
            "mode": "stream", "provider": self.provider, "prompt": prompt,
            "summary": summary.to_dicts(), "previous_codebook": book,
        })
        if note is not None:
            (folder / "note.txt").write_text(note, encoding="utf-8")
        if (folder / "codebook_response.json").exists():
            return save_codebook(folder)
        kwargs = dict(coded_documents=summary, prompt=prompt, model=self.models["A"],
                      api_key=self.api_key, out_dir=folder, dry_run=preview,
                      provider=self.provider)
        if book is None:
            path = generate_codebook(**kwargs)
        else:
            path = revise_codebook(**kwargs, previous_codebook=book)
        print(f"{'Preview ready' if preview else 'Saved'}: codebook version {version} (stream).")
        return path

    def show_codebook(self, version):
        """Return the saved path; also extract responses saved by older code."""
        return save_codebook(self._version_dir(version))

    def compare(self, datasets, version):
        folders = [self._coding_dir(datasets, version, model) for model in ["A", "B"]]
        for folder in folders:
            if self._results(folder) is None:
                return
        compare_models(*folders, n_runs=self.n_runs)

    def accept(self, version, note):
        """Record your decision to keep a version; no copy or new version is created."""
        self._book(version)
        path = self._version_dir(version) / "accepted.json"
        path.write_text(json.dumps({"version": version, "note": note}, indent=2), encoding="utf-8")
        print(f"Accepted version {version}: {note}")

    def status(self):
        """Show saved task states without network requests."""
        rows = []
        for path in sorted(self.root.glob("**/task.json")):
            folder = path.parent
            task = json.loads(path.read_text(encoding="utf-8"))
            if (folder / "batch.json").exists():
                state = json.loads((folder / "batch.json").read_text()).get("status", "submitted")
            elif task["kind"] == "codebook" and (folder / "codebook_response.json").exists():
                state = "completed"
            else:
                count = len(list(folder.glob("run_[0-9][0-9].json")))
                state = "completed" if count == task.get("n_runs", 0) - len(_excluded_runs(folder)) else f"{count} runs saved" if count else "preview"
            if (folder / "accepted.json").exists():
                state = "accepted"
            rows.append({"task": str(folder.relative_to(self.root)), "state": state})
        if not rows:
            print("No tasks yet.")
        else:
            print(pl.DataFrame(rows))


# Helpers: prepare requests, save responses, and draw review plots.


def load_documents(path):
    """Read a CSV and return its document_id and text columns."""
    return pl.read_csv(path, columns=["document_id", "text"], infer_schema=False)


def build_classification_request(documents, prompt, codebook=None):
    """Build the request shared by single calls and batch runs."""
    item = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["0", "1"]},
            "explanation": {"type": "string"},
        },
        "required": ["label", "explanation"],
        "additionalProperties": False,
    }
    if codebook is not None:
        item["properties"]["citation"] = {"type": "integer"}
        item["required"].append("citation")
        prompt += "\n\nCodebook:\n" + json.dumps(codebook, ensure_ascii=False)
    document_ids = documents["document_id"].to_list()
    schema = {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "object",
                "properties": {document_id: item for document_id in document_ids},
                "required": document_ids,
                "additionalProperties": False,
            },
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(documents.to_dicts())},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "classifications", "strict": True, "schema": schema},
        },
        "stream": False,
    }


def parse_classifications(response):
    """Convert a classification response to a Polars table."""
    content = response["choices"][0]["message"]["content"]
    classifications = json.loads(content)["classifications"]
    if isinstance(classifications, dict):
        classifications = [
            {**values, "document_id": document_id}
            for document_id, values in classifications.items()
        ]
    table = pl.DataFrame(classifications)
    return table.select(sorted(table.columns))


def _excluded_runs(folder):
    path = Path(folder) / "excluded_runs.json"
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def send_requests(payload, api_key, folder, names, mode, preview, request_file):
    """Save a preview, or send requests and reuse any answers already saved."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload["stream"] = mode == "stream"
    if mode == "batch":
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": payload["model"],
            "completion_window": "24h",
            "requests": [{"custom_id": name, "body": payload} for name in names],
        }
    (folder / request_file).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if preview:
        return
    if mode == "batch":
        if (folder / "batch.json").exists():
            return
        response = requests.post(
            "https://openrouter.ai/api/v1/batches",
            headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=120,
        )
        response.raise_for_status()
        (folder / "batch.json").write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
        return
    with OpenRouter(api_key=api_key) as client:
        for number, name in enumerate(names, start=1):
            if name in _excluded_runs(folder):
                continue
            filename = "codebook_response.json" if name == "codebook" else f"{name}.json"
            path = folder / filename
            if path.exists():
                continue
            for attempt in range(4):
                response = send_stream(client, payload)
                try:
                    json.loads(response["choices"][0]["message"]["content"])
                    break
                except json.JSONDecodeError as error:
                    if attempt < 3:
                        print(f"{name}: invalid JSON; retry {attempt + 1}/3.")
                        continue
                    reason = response["choices"][0].get("finish_reason", "unknown")
                    raise ValueError(
                        f"{name} returned invalid JSON after 4 attempts (finish reason: {reason}). "
                        "Rerun the cell to retry."
                    ) from error
            path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
            if name != "codebook":
                print(f"Run {number}/{len(names)}")


def retrieve_classification_batch(api_key, out_dir):
    """Check status once and save the batch, including raw results when ready."""
    path = Path(out_dir) / "batch.json"
    batch = json.loads(path.read_text(encoding="utf-8"))
    response = requests.get(
        f"https://openrouter.ai/api/v1/batches/{batch['id']}",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=120,
    )
    response.raise_for_status()
    batch = response.json()
    path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    return batch


def send_stream(client, payload):
    """Assemble one streamed response."""
    message = {"role": "assistant", "content": "", "reasoning": ""}
    choice = {"message": message}
    output = {"choices": [choice]}
    for chunk in client.chat.send(**payload, timeout_ms=120000):
        data = chunk.model_dump(mode="json", by_alias=True, exclude_none=True)
        output.update({key: value for key, value in data.items() if key != "choices"})
        for item in data.get("choices", []):
            delta = item["delta"]
            for key in ("content", "reasoning"):
                message[key] += delta.get(key) or ""
            if delta.get("reasoning_details"):
                message.setdefault("reasoning_details", []).extend(delta["reasoning_details"])
            if item.get("finish_reason"):
                choice["finish_reason"] = item["finish_reason"]
    output["object"] = "chat.completion"
    return output


def parse_classification_runs(out_dir, mode=None):
    """Parse saved batch or streaming responses together and save a CSV."""
    out_dir = Path(out_dir)
    mode = mode or ("batch" if (out_dir / "batch.json").exists() else "stream")
    runs = []
    if mode == "batch":
        batch = json.loads((out_dir / "batch.json").read_text(encoding="utf-8"))
        expected = set(load_documents(out_dir / "documents.csv")["document_id"].to_list())
        skipped = []
        for item in batch.get("results", []):
            try:
                run = int(item["custom_id"].split("_")[1])
                classifications = parse_classifications(item["response"]["body"])
                if (classifications.height != len(expected)
                        or set(classifications["document_id"].to_list()) != expected
                        or not set(classifications["label"].to_list()) <= {"0", "1"}
                        or "explanation" not in classifications.columns):
                    raise ValueError("Incomplete or invalid classifications")
                runs.append(classifications.with_columns(pl.lit(run).alias("run")))
            except (ValueError, KeyError, TypeError, pl.exceptions.PolarsError):
                skipped.append(item.get("custom_id", "unknown"))
        if skipped or batch.get("request_counts", {}).get("failed", 0):
            print(f"Using {len(runs)} successful batch runs; omitted failed runs" +
                  (f": {', '.join(skipped)}." if skipped else "."))
        if not runs:
            raise ValueError("No successful batch runs to review.")
    else:
        for path in sorted(out_dir.glob("run_*.json")):
            if path.stem in _excluded_runs(out_dir):
                continue
            if path.name.endswith(".in_progress.json"):
                continue
            response = json.loads(path.read_text(encoding="utf-8"))
            classifications = parse_classifications(response)
            run = int(path.stem.split("_")[1])
            runs.append(classifications.with_columns(pl.lit(run).alias("run")))
    results = pl.concat(runs).unique(maintain_order=True).sort("run", "document_id")
    results.write_csv(out_dir / "classifications.csv")
    return results


def codebook_input(documents, results):
    """Summarize each document with its coding percentage and example rationales."""
    summary = results.group_by("document_id").agg(
        ((pl.col("label") == "1").mean() * 100).alias("percent_1"),
        pl.col("explanation").filter(pl.col("label") == "1").first().alias("explanation_1"),
        pl.col("explanation").filter(pl.col("label") == "0").first().alias("explanation_0"),
    )
    return documents.join(summary, on="document_id", how="left")


def generate_codebook(coded_documents, prompt, model, api_key, out_dir, dry_run=False, provider=None, previous_codebook=None):
    """Preview or submit a request to generate a codebook."""
    return _request_codebook(coded_documents, prompt, model, api_key, out_dir,
                             dry_run, provider, previous_codebook)


def revise_codebook(coded_documents, prompt, model, api_key, out_dir,
                    previous_codebook, dry_run=True, provider=None):
    """Preview or submit a revision with rule statuses and change documentation."""
    rule_schema = {
        "type": "object",
        "properties": {
            "number": {"type": "integer"},
            "rule": {"type": "string"},
            "status": {"type": "string", "enum": ["revised", "unchanged", "created"]},
        },
        "required": ["number", "rule", "status"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "codebook": {
                "type": "object",
                "properties": {
                    "definition": {"type": "string"},
                    "decision_rules": {"type": "array", "items": rule_schema,
                                       "minItems": 1},
                },
                "required": ["definition", "decision_rules"],
                "additionalProperties": False,
            },
            "documentation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "description": {"type": "string"},
                    },
                    "required": ["number", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["codebook", "documentation"],
        "additionalProperties": False,
    }
    return _request_codebook(coded_documents, prompt, model, api_key, out_dir,
                             dry_run, provider, previous_codebook, schema)


def _request_codebook(coded_documents, prompt, model, api_key, out_dir, dry_run=False,
                      provider=None, previous_codebook=None, response_schema=None):
    """Preview or send one codebook request and save the raw response."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if previous_codebook is not None:
        prompt += "\n\nCurrent codebook:\n" + json.dumps(previous_codebook, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(coded_documents.to_dicts(), ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "codebook", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "definition": {"type": "string"},
                        "decision_rules": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "number": {"type": "integer"},
                                    "rule": {"type": "string"},
                                },
                                "required": ["number", "rule"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["definition", "decision_rules"],
                    "additionalProperties": False,
                },
            },
        },
        "stream": False,
    }
    if response_schema is not None:
        payload["response_format"]["json_schema"].update(
            name="codebook_revision", schema=response_schema,
        )
    if provider:
        payload["provider"] = {"only": [provider]}
    send_requests(payload, api_key, out_dir, ["codebook"], "stream", dry_run, "codebook_request.json")
    if not dry_run:
        return save_codebook(out_dir)


def save_codebook(folder):
    """Extract a streamed response into the codebook and optional revision notes."""
    folder = Path(folder)
    response = json.loads((folder / "codebook_response.json").read_text(encoding="utf-8"))
    data = json.loads(response["choices"][0]["message"]["content"])
    path = folder / "codebook.json"
    path.write_text(json.dumps(data.get("codebook", data), indent=2, ensure_ascii=False), encoding="utf-8")
    if "documentation" in data:
        (folder / "documentation.json").write_text(json.dumps(data["documentation"], indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_classifications(documents, prompt, model, api_key, out_dir, mode="stream", n_runs=25, dry_run=True, codebook=None, provider=None):
    """Save the input documents and send the repeated classification request."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents.write_csv(out_dir / "documents.csv")
    payload = {"model": model, **build_classification_request(documents, prompt, codebook)}
    if provider:
        payload["provider"] = {"only": [provider]}
    names = [f"run_{run:02d}" for run in range(1, n_runs + 1)]
    request_file = "batch_request.json" if mode == "batch" else "stream_request.json"
    send_requests(payload, api_key, out_dir, names, mode, dry_run, request_file)


def show_agreement(results, document_order=None):
    """Print nominal alpha and plot majority agreement; return IDs in plot order.

    By default, sort by agreement then document ID. Supply the returned list to
    reuse its order for another figure containing the same document IDs.
    """
    ratings = results.with_columns(pl.col("label").cast(pl.Float64)).pivot(
        on="run", index="document_id", values="label",
    ).drop("document_id").to_numpy().T
    alpha = krippendorff.alpha(
        reliability_data=ratings, level_of_measurement="nominal", value_domain=[0, 1],
    ) if ratings.shape[0] > 1 and results["label"].n_unique() > 1 else float("nan")
    agreement = results.group_by("document_id").agg(
        (pl.col("label") == "1").mean().alias("p_1"),
    ).with_columns(
        pl.max_horizontal("p_1", 1 - pl.col("p_1")).alias("agreement"),
    ).sort("agreement", "document_id")
    if document_order is not None:
        document_order = list(document_order)
        order = pl.DataFrame({"document_id": document_order}).with_row_index("_position")
        agreement = order.join(agreement, on="document_id", how="left").sort("_position")
    document_order = agreement["document_id"].to_list()
    print(f"Krippendorff's alpha: {alpha:.4f}")
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = list(range(agreement.height))
    ax.scatter(positions, agreement["agreement"].to_list(), color="black", marker="D", s=12)
    ax.set_xticks(positions, agreement["document_id"].to_list(), rotation=90, fontsize=6)
    ax.set(xlabel="document_id", ylabel="agreement", ylim=(0.48, 1.02))
    ax.set_axisbelow(True)
    ax.grid(True, color="0.9", linewidth=0.5)
    fig.tight_layout()
    plt.show()
    return document_order


def compare_models(a_dir, b_dir, n_runs=25):
    """Compare label-1 percentages and nominal alpha for matching A/B run IDs.

    Use matching documents and complete runs in each model.
    Undefined correlation or any undefined run alpha is reported as NaN.
    Print metrics and display the plot without writing files.
    """
    frames = []
    for name, directory in [("A", a_dir), ("B", b_dir)]:
        frame = pl.read_csv(
            Path(directory) / "classifications.csv",
            schema_overrides={"document_id": pl.String, "label": pl.String},
        ).select("document_id", "run", "label")
        frames.append(frame)
    a, b = frames
    percentages = [
        frame.group_by("document_id").agg(
            ((pl.col("label") == "1").mean()).alias(f"percent_1_{name}")
        ) for name, frame in zip(["A", "B"], frames)
    ]
    comparison = percentages[0].join(percentages[1], on="document_id").with_columns(
        (pl.col("percent_1_A") - pl.col("percent_1_B")).abs().alias("difference_pp"),
    ).sort("difference_pp", "document_id", descending=[True, False])
    paired = a.rename({"label": "label_A"}).join(
        b.rename({"label": "label_B"}), on=["document_id", "run"],
    ).sort("run", "document_id")
    alphas = []
    for run in sorted(paired["run"].unique().to_list()):
        ratings = paired.filter(pl.col("run") == run).select(
            pl.col("label_A").cast(pl.Float64), pl.col("label_B").cast(pl.Float64),
        ).to_numpy().T
        alpha = krippendorff.alpha(
            reliability_data=ratings, level_of_measurement="nominal", value_domain=[0, 1],
        ) if len(set(ratings.flatten())) > 1 else float("nan")
        alphas.append({"run": run, "alpha": alpha})
    run_alphas = pl.DataFrame(alphas, schema={"run": pl.Int64, "alpha": pl.Float64})
    mean_difference = comparison["difference_pp"].mean()
    correlation = comparison.select(pl.corr("percent_1_A", "percent_1_B")).item()
    mean_alpha = run_alphas["alpha"].mean()
    sd_alpha = run_alphas["alpha"].std(ddof=1) if run_alphas.height > 1 else float("nan")
    metrics = pl.DataFrame({
        "n_documents": [comparison.height],
        "n_runs_A": [a["run"].n_unique()], "n_runs_B": [b["run"].n_unique()],
        "n_paired_runs": [run_alphas.height],
        "mean_absolute_difference_pp": [mean_difference],
        "pearson_correlation": [correlation],
        "mean_pairwise_alpha": [mean_alpha],
        "sd_pairwise_alpha": [sd_alpha],
    })
    print(f"Mean abs difference in label-1 proportions: {mean_difference:.4f}")
    print(f"Model correlation: {correlation:.4f}")
    print(f"Mean paired-run alpha: {mean_alpha:.4f}")
    print(f"SD of paired-run alpha: {sd_alpha:.4f}")
    print("Paired-run alphas:")
    for run, alpha in run_alphas.iter_rows():
        print(f"Run {run:02d}: {alpha:.4f}")
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(0)
    x = comparison["percent_1_A"].to_numpy() + rng.uniform(-0.005, 0.005, comparison.height)
    y = comparison["percent_1_B"].to_numpy() + rng.uniform(-0.005, 0.005, comparison.height)
    ax.scatter(x, y, color="black", s=20, alpha=0.7)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.set(xlabel=f"Proportion of {a['run'].n_unique()} A runs coded 1", ylabel=f"Proportion of {b['run'].n_unique()} B runs coded 1",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show()
    ordered = comparison.sort("percent_1_A", "document_id", descending=[True, False])
    positions = list(range(ordered.height))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(positions, ordered["percent_1_A"].to_numpy() * 100,
               color="tab:blue", label="Model A", s=20)
    ax.scatter(positions, ordered["percent_1_B"].to_numpy() * 100,
               color="tab:orange", label="Model B", marker="x", s=25)
    ax.set_xticks(positions, ordered["document_id"].to_list(), rotation=90, fontsize=6)
    ax.set(xlabel="Document ID (ordered by model A, highest to lowest)",
           ylabel="Runs coded 1 (%)", ylim=(-2, 102))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return comparison, metrics

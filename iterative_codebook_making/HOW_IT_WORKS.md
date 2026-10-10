# How the codebook workflow works

Use `simple_codebook_workflow.ipynb` to run the experiment. Its implementation is in one file, `simple_experiment.py`. The older `codebook_iteration_funcs.py` exists only to support the original notebook; the simple workflow does not import it.

## Setup

Start the notebook kernel in `iterative_codebook_making`, alongside the Python file.

```python
import os
from dotenv import load_dotenv
from simple_experiment import Experiment

load_dotenv("../.env")

experiment = Experiment(
    session_dir="emotive_rhetoric",
    model_a="openai/gpt-6-luna",
    model_b="deepseek/deepseek-v4.1-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    mode="stream",
    n_runs=25,
)
```

`import` makes code from another file available. `load_dotenv` loads your key from the `.env` file one folder above the working directory. `os.getenv` reads that key.

`Experiment` is a class: a way to keep settings and related functions together. `experiment` is the particular experiment you created. Creating it saves the settings in memory; it does not call a model or start coding. Inside the class, `self` refers to this experiment. For example, `self.n_runs` is its run count.

## Files

```text
emotive_rhetoric/
  data/
    emotive_rhetoric_0.csv
    emotive_rhetoric_1.csv
  prompts/
    initial_classification.txt
    classify_w_codebook.txt
    generate_A00.txt
    improve_codebook.txt
  out/
    coding/
    codebooks/
    orders/
```

Input CSV files need `document_id` and `text` columns. Document IDs must be unique, including when you combine datasets. The prompt files contain the instructions sent to the model. The code reads these files and creates output folders as needed.

`datasets=0` selects `emotive_rhetoric_0.csv`. `datasets=[0, 1]` combines files 0 and 1. A dataset number identifies input data; a version number identifies a codebook.

## The steps

1. `experiment.code(datasets=0, preview=True)` prepares a request and saves it without contacting the model. Change to `preview=False` to send it. With `n_runs=25`, it sends the same set of documents 25 times, collecting labels and explanations.
2. `experiment.review(datasets=0)` reads the saved answers, prints Krippendorff's alpha, and plots agreement for each document. It checks batch status when needed, but never submits new work.
3. `experiment.generate(datasets=0, version=1, preview=False)` uses model A to write a codebook from the initial results. Each document is summarized with its proportion of label-1 answers and example explanations.
4. `experiment.show_codebook(version=1)` prints the definition and rules and saves `codebook.json`. Run this before using that version for coding.
5. `experiment.code(datasets=0, version=1, preview=False)` repeats coding with the codebook included in the instructions. Review it with `experiment.review(datasets=0, version=1)`.
6. If necessary, `experiment.revise(datasets=0, source_version=1, version=2, preview=False)` uses results from version 1 to write version 2. Show version 2, then code and review again. You decide whether another revision is needed.

The notebook defaults to previews. In the examples above, `preview=False` sends requests that use your API account. You can preview every request first.

`version=None`, the default, means coding without a codebook. Model A is the default model. `model="B"` selects your second model.

## Optional steps

`experiment.compare(datasets=0, version=1)` compares A and B after both have coded the same documents with that version. It prints three statistics and displays a scatterplot.

`additions_only=True` tells a revision to preserve existing rules and append new ones. `compare_models=True` includes coding summaries from both models in the revision request. These are instructions to the model, not guarantees enforced by Python.

`experiment.accept(version=1, note="Ready to use")` records your decision. It does not automatically select that version for future calls; you still supply `version=1`.

`experiment.status()` lists saved tasks and their progress. It reads local files and does not call the API.

## Saving and resuming

Streaming mode sends requests now and saves one response per run. Batch mode submits a group of requests; rerun the review cell later to check whether they are ready. Neither mode automatically moves to the next stage.

Rerunning a coding cell skips saved streaming runs or reuses the saved batch. Codebook requests also reuse their saved response or batch. Restarting the kernel loses Python's in-memory variables, but leaves the saved files intact. Rerun setup and use the same session and settings to continue. After editing the Python module, restart the kernel so imports use the new code.

The remaining checks prevent mixing changed inputs with saved results, reviewing incomplete coding, and overwriting a source codebook with its revision. Ordinary Python or API errors handle other problems. There are no automatic response retries: if a request fails, fix the issue and rerun the cell. A saved malformed response must be inspected and removed before resubmitting that run.

## Reading the Python file

Functions below `class Experiment` do the background work: build requests, send them, read responses, and draw plots. Methods inside `Experiment` connect those functions into notebook steps. Start reading `code`, `review`, `generate`, and `revise`; the JSON schemas are just instructions describing the answer format required from the model.

A method name starting with `_`, such as `_prompt`, is a helper used inside the class. You normally do not call it from the notebook. `Path` represents a file or folder, and `/` joins paths: `self.session_dir / "out"` means the `out` folder inside your session.

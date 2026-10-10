# Running the codebook procedure

For the simpler workflow, open `simple_codebook_workflow.ipynb`. For a beginner explanation, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md). It uses the standalone `simple_experiment.py` to manage task paths, request reuse, batch checks, and saved plot order. Codebook versions increase continuously (1, 2, 3…) independently of dataset numbers. Each `session_dir` identifies one experiment, with its results saved under `<session>/out/`.

Open `codebook_iteration.ipynb` from this folder. Set `session` at the top to group one experiment. Each task has its own mode and provider. Run its configuration cell before constructing its output path. All submission cells start with `dry_run = True`; change it only after reviewing the request. Retrieve batch jobs until completed, then run the separate parsing cell.

The notebook follows stages I–III of `codebook_iteration_procedure.txt`. Branches are manual: inspect agreement, decide whether to freeze or revise, and rerun the relevant section with the next dataset or codebook revision. The procedure does not specify an acceptance threshold. Revision sections are optional and should be skipped when agreement is sufficient.

## Folder layout

```text
out/<session>/
  I_initial/dataset_0/uncoded/<model>/<mode>/
  codebooks/A.0.0/<model>/<mode>/
  I_consistency/dataset_0/A.0.k/<model>/<mode>/
  II_new_data/dataset_j/<previous_codebook>/<model>/<mode>/
  codebooks/A.j.k/<model>/<mode>/
  II_consistency/datasets_0_to_j/A.j.k/<model>/<mode>/
  III_cross_model/datasets_<numbers>/Ab.j.k/<model>/<mode>/
  III_comparisons/datasets_<numbers>/Ab.j.k/
  codebooks/Ab.j.k/<model_A>/<mode>/
  frozen/
```

Provider choice is recorded in `settings.json` and the request. If comparing providers, use a new session or add a provider component to the task directory. Completed outputs are reused, so use a new directory when inputs change. Existing outputs from before this organization remain in their original locations.

## Classification tasks

Each folder contains the input `documents.csv`, `settings.json`, and either streaming request/responses or batch request/job/results. `review_coding(path)` parses the saved responses, writes only `classifications.csv`, prints nominal Krippendorff alpha across runs, and displays a sorted per-document majority-agreement figure. It returns the document IDs in plot order. Capture that list and pass it as `document_order` to reuse the order for another figure with the same document IDs:

```python
order_0 = review_coding(dir_0)
review_coding(A0K_dir_consistency, document_order=order_0);

# Save the order across kernel restarts.
Path("document_order.json").write_text(json.dumps(order_0), encoding="utf-8")
order_0 = json.loads(Path("document_order.json").read_text(encoding="utf-8"))
```

`show_agreement(results, document_order=None)` also returns the plotted IDs. A supplied order must contain each document ID exactly once. Assign the return value or end a notebook call with `;` to suppress display of the list. Wide tables and codebook summaries can be built separately when needed:

- `classifications.csv`: one row per document per run, including explanations and citations when applicable.
- `wide`: texts and IDs with one label/explanation column per run.
- `summary`: texts, IDs, percentage coded 1, and one explanation per observed label.

`review_coding` does not require `documents.csv`. To build wide tables or codebook summaries for an older classification directory, use a snapshot of the exact input documents. Do not guess the input dataset from an old folder name.

## Codebook tasks and acceptance

Generation folders contain `codebook_request.json` and either `codebook_response.json` or `batch.json`. `parse_codebook(path, codebook_id="A.1.0")` writes `codebook_A.1.0.json`. It sends no requests. The request stores the supplied coding summaries and the previous codebook when revising.

Use `generate_codebook` for initial generation and `revise_codebook` for revisions:

```python
revision_response = revise_codebook(
    coded_documents=revision_docs, prompt=revise_prompt,
    previous_codebook=codebook_A0K, model=model_a, api_key=api_key,
    out_dir=A0K_revised_dir, dry_run=True,
)
# After reviewing the payload, rerun with dry_run=False.
# For batch mode, retrieve the completed batch before parsing.
codebook_A0K_revise = parse_codebook(A0K_revised_dir, codebook_id=revised_codebook_name)
```

Revision responses contain `codebook` (definition and full decision rules with `number`, `rule`, and `status`) and `documentation` (affected rule `number` and change `description`). Status is `revised`, `unchanged`, or `created`, relative to the supplied codebook. The prompt requests documentation only for revised or created rules; the schema enforces the fields and status values, but not correspondence between the two lists. Parsing saves `documentation_<version>.json` separately and prints the codebook and returns its saved Path. Load its contents with json.loads(path.read_text(encoding="utf-8")) for subsequent classification or revision calls; return_path=False retains the dictionary return when needed. Initial-generation responses retain their existing format.

`freeze_codebook(source, destination, frozen_id, note)` copies a manually accepted codebook and saves the source and rationale in `decision.json`. Use a separate destination directory per freeze decision if you want to retain every decision record.

For stage II, test each revision against all datasets through j. Stage III uses the same codebook and documents for both models. `compare_models(a_dir, b_dir, n_runs=25)` prints mean absolute difference in label-1 percentages (percentage points), Pearson correlation of those percentages across documents, and mean nominal Krippendorff alpha across matching A/B run IDs. Each run alpha treats A and B as two coders across all documents. Both models must have the same documents and run IDs, with exactly one classification per document/run. Constant percentages yield undefined correlation; if any run alpha is undefined (all pooled ratings have one label), its NaN propagates into the average. The function displays a scatterplot with A percentages on x, B percentages on y, and a y=x line. It writes no analysis files and returns `(comparison, metrics)`.

All analysis is based on returned classifications. Counts in the saved tables expose incomplete runs; no automatic repair or extra validation is applied.

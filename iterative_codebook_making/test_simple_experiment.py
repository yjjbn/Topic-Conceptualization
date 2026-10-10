"""Offline checks for task reuse and the notebook workflow."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import polars as pl

from simple_experiment import Experiment
import simple_experiment as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = Path(self.temp.name) / "example"
        (self.session / "data").mkdir(parents=True)
        (self.session / "prompts").mkdir()
        for name in ["initial_classification.txt", "classify_w_codebook.txt",
                     "generate_A00.txt", "improve_codebook.txt"]:
            (self.session / "prompts" / name).write_text("Test prompt")
        pl.DataFrame({"document_id": ["001", "002"], "text": ["one", "two"]}).write_csv(
            self.session / "data/example_0.csv")
        self.experiment = Experiment(self.session, "model/a", "model/b", "test", n_runs=2)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        workflow.plt.close("all")
        self.output.__exit__(None, None, None)
        self.temp.cleanup()

    def responses(self):
        self.experiment.code(0)
        folder = self.experiment._coding_dir(0, None, "A")
        response = {"choices": [{"message": {"content": json.dumps({
            "classifications": {"001": {"label": "0", "explanation": "no"},
                                "002": {"label": "1", "explanation": "yes"}}
        })}}]}
        for run in [1, 2]:
            (folder / f"run_{run:02d}.json").write_text(json.dumps(response))

    def test_preview_and_changed_inputs(self):
        with patch.object(workflow, "OpenRouter", side_effect=AssertionError("No API in preview")):
            self.experiment.code(0)
        self.experiment.code(0)
        self.experiment.models["A"] = "different/model"
        with self.assertRaisesRegex(ValueError, "different inputs"):
            self.experiment.code(0)

    def test_review_order_survives_restart(self):
        self.responses()
        with patch.object(workflow.plt, "show"):
            self.experiment.review(0)
        order_path = self.experiment.root / "orders/datasets-0.json"
        self.assertEqual(json.loads(order_path.read_text()), ["001", "002"])
        restarted = Experiment(self.session, "model/a", "model/b", "test", n_runs=2)
        with patch.object(workflow.plt, "show"):
            restarted.review(0)

    def test_streaming_saves_runs_and_reuses_them(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "classifications": {"001": {"label": "0", "explanation": "no"},
                                "002": {"label": "1", "explanation": "yes"}}
        })}}]}
        with patch.object(workflow, "OpenRouter"), \
             patch.object(workflow, "send_stream", return_value=response) as send:
            self.experiment.code(0, preview=False)
            self.assertEqual(send.call_count, 2)
            self.experiment.code(0, preview=False)
            self.assertEqual(send.call_count, 2)
        self.assertEqual(self.experiment._results(
            self.experiment._coding_dir(0, None, "A")).height, 4)

    def test_generation_and_revision_no_duplicate_submission(self):
        self.responses()
        self.experiment.generate()
        folder = self.experiment._version_dir(1)
        book = {"definition": "Test", "decision_rules": [{"number": 1, "rule": "Test rule"}]}
        response = {"choices": [{"message": {"content": json.dumps(book)}}]}
        (folder / "codebook_response.json").write_text(json.dumps(response))
        self.assertEqual(self.experiment.show_codebook(1), folder / "codebook.json")
        with patch.object(workflow, "generate_codebook") as submit:
            self.experiment.generate(preview=False)
            submit.assert_not_called()
        self.experiment.code(0, version=1)
        source = self.experiment._coding_dir(0, None, "A")
        target = self.experiment._coding_dir(0, 1, "A")
        for run in [1, 2]:
            (target / f"run_{run:02d}.json").write_text((source / f"run_{run:02d}.json").read_text())
        self.experiment.revise(0, source_version=1, version=2)
        revised = {"codebook": book, "documentation": [{"number": 1, "description": "Clarified"}]}
        response["choices"][0]["message"]["content"] = json.dumps(revised)
        (self.experiment._version_dir(2) / "codebook_response.json").write_text(json.dumps(response))
        self.experiment.show_codebook(2)
        self.assertTrue((self.experiment._version_dir(2) / "documentation.json").exists())
        self.experiment.accept(2, "Ready")
        self.assertTrue((self.experiment._version_dir(2) / "accepted.json").exists())
        with self.assertRaises(ValueError):
            self.experiment.revise(0, source_version=2, version=2)

    def test_pending_batch_review_never_submits(self):
        self.experiment.mode = "batch"
        self.experiment.code(0)
        folder = self.experiment._coding_dir(0, None, "A")
        (folder / "batch.json").write_text(json.dumps({"id": "test", "status": "pending"}))
        with patch.object(workflow, "retrieve_classification_batch", return_value={"status": "pending"}), \
             patch.object(workflow, "run_classifications") as submit:
            self.experiment.review(0)
            submit.assert_not_called()

    def test_comparison_and_cross_model_revision(self):
        self.responses()
        folder = self.experiment._version_dir(1)
        folder.mkdir(parents=True)
        self.experiment.codebook_path(1).write_text(json.dumps({
            "definition": "Test", "decision_rules": [{"number": 1, "rule": "Test"}],
        }))
        source = self.experiment._coding_dir(0, None, "A")
        for model in ["A", "B"]:
            self.experiment.code(0, version=1, model=model)
            target = self.experiment._coding_dir(0, 1, model)
            for run in [1, 2]:
                (target / f"run_{run:02d}.json").write_text((source / f"run_{run:02d}.json").read_text())
        with patch.object(workflow.plt, "show"):
            self.experiment.compare(0, version=1)
        self.experiment.revise(0, source_version=1, version=2, compare_models=True)
        request = json.loads((self.experiment._version_dir(2) / "codebook_request.json").read_text())
        documents = json.loads(request["messages"][1]["content"])
        self.assertIn("percent_1_B", documents[0])
        self.assertEqual(documents[0]["percent_1"], documents[0]["percent_1_B"])
        self.assertFalse(list(self.experiment.root.glob("**/model_comparison*")))

    def test_incomplete_runs_rejected(self):
        self.responses()
        folder = self.experiment._coding_dir(0, None, "A")
        (folder / "run_02.json").write_text(json.dumps({"choices": [{"message": {"content": json.dumps({
            "classifications": {"001": {"label": "0", "explanation": "no"}}
        })}}]}))
        with self.assertRaisesRegex(ValueError, "all input document IDs"):
            self.experiment.review(0)


if __name__ == "__main__":
    unittest.main()

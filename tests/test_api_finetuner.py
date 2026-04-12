"""Tests for socialscikit.quantikit.api_finetuner."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from socialscikit.quantikit.api_finetuner import (
    APIFineTuneConfig,
    APIFineTuneResult,
    APIFineTuneStatus,
    APIFineTuner,
    FINETUNE_MODELS,
    MIN_EXAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(n: int = 20) -> pd.DataFrame:
    texts = [f"sample text {i}" for i in range(n)]
    labels = ["positive" if i % 2 == 0 else "negative" for i in range(n)]
    return pd.DataFrame({"text": texts, "label": labels})


def _mock_job(status="succeeded", model_id="ft:gpt-4o-mini:org::abc123"):
    job = MagicMock()
    job.id = "ftjob-test123"
    job.status = status
    job.training_file = "file-abc"
    job.fine_tuned_model = model_id if status == "succeeded" else None
    job.trained_tokens = 5000 if status == "succeeded" else None
    job.error = MagicMock()
    job.error.message = None if status != "failed" else "data format error"
    job.created_at = 1700000000
    job.finished_at = 1700003600 if status == "succeeded" else None
    return job


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):

    def test_config_defaults(self):
        c = APIFineTuneConfig()
        self.assertEqual(c.model, "gpt-4o-mini-2024-07-18")
        self.assertEqual(c.n_epochs, "auto")
        self.assertEqual(c.batch_size, "auto")
        self.assertEqual(c.seed, 42)

    def test_status_fields(self):
        s = APIFineTuneStatus(job_id="j1", status="running", file_id="f1")
        self.assertEqual(s.job_id, "j1")
        self.assertIsNone(s.fine_tuned_model)

    def test_result_fields(self):
        r = APIFineTuneResult(fine_tuned_model="ft:model", job_id="j1")
        self.assertEqual(r.predictions, [])

    def test_finetune_models_list(self):
        self.assertIn("gpt-4o-mini-2024-07-18", FINETUNE_MODELS)
        self.assertTrue(len(FINETUNE_MODELS) >= 2)


# ---------------------------------------------------------------------------
# JSONL preparation
# ---------------------------------------------------------------------------

class TestPrepareJsonl(unittest.TestCase):

    def setUp(self):
        self.ft = APIFineTuner(api_key="sk-test")
        self.df = _sample_df(20)

    def test_basic_jsonl(self):
        path = self.ft.prepare_jsonl(self.df, "text", "label", ["positive", "negative"])
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 20)
        first = json.loads(lines[0])
        self.assertIn("messages", first)
        self.assertEqual(len(first["messages"]), 3)
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertEqual(first["messages"][1]["role"], "user")
        self.assertEqual(first["messages"][2]["role"], "assistant")
        os.unlink(path)

    def test_label_in_expected_set(self):
        path = self.ft.prepare_jsonl(self.df, "text", "label", ["positive", "negative"])
        with open(path) as f:
            for line in f:
                msg = json.loads(line)
                label = msg["messages"][2]["content"]
                self.assertIn(label, ["positive", "negative"])
        os.unlink(path)

    def test_with_definitions(self):
        defs = {"positive": "good", "negative": "bad"}
        path = self.ft.prepare_jsonl(
            self.df, "text", "label", ["positive", "negative"],
            label_definitions=defs,
        )
        with open(path) as f:
            first = json.loads(f.readline())
        sys_content = first["messages"][0]["content"]
        self.assertIn("good", sys_content)
        self.assertIn("bad", sys_content)
        os.unlink(path)

    def test_too_few_examples(self):
        small_df = _sample_df(5)
        with self.assertRaises(ValueError) as ctx:
            self.ft.prepare_jsonl(small_df, "text", "label", ["positive", "negative"])
        self.assertIn("10", str(ctx.exception))

    def test_unknown_labels(self):
        with self.assertRaises(ValueError) as ctx:
            self.ft.prepare_jsonl(self.df, "text", "label", ["positive"])
        self.assertIn("未定义", str(ctx.exception))

    def test_stores_system_prompt(self):
        path = self.ft.prepare_jsonl(self.df, "text", "label", ["positive", "negative"])
        self.assertIn("positive", self.ft._system_prompt)
        os.unlink(path)

    def test_drops_na_rows(self):
        df = _sample_df(15)
        df.loc[0, "label"] = None
        path = self.ft.prepare_jsonl(df, "text", "label", ["positive", "negative"])
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 14)
        os.unlink(path)


# ---------------------------------------------------------------------------
# Mocked API calls
# ---------------------------------------------------------------------------

class TestUploadFile(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_upload(self, mock_get):
        client = MagicMock()
        resp = MagicMock()
        resp.id = "file-xyz"
        client.files.create.return_value = resp
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        # Create a dummy file
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write('{"messages":[]}\n')
        tmp.close()

        file_id = ft.upload_file(tmp.name)
        self.assertEqual(file_id, "file-xyz")
        client.files.create.assert_called_once()
        os.unlink(tmp.name)


class TestCreateJob(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_create_default(self, mock_get):
        client = MagicMock()
        job = MagicMock()
        job.id = "ftjob-abc"
        client.fine_tuning.jobs.create.return_value = job
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        job_id = ft.create_job("file-xyz")
        self.assertEqual(job_id, "ftjob-abc")
        call_kwargs = client.fine_tuning.jobs.create.call_args[1]
        self.assertEqual(call_kwargs["training_file"], "file-xyz")
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini-2024-07-18")

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_create_with_config(self, mock_get):
        client = MagicMock()
        job = MagicMock()
        job.id = "ftjob-def"
        client.fine_tuning.jobs.create.return_value = job
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        config = APIFineTuneConfig(n_epochs="3", suffix="test")
        job_id = ft.create_job("file-xyz", config)
        self.assertEqual(job_id, "ftjob-def")
        call_kwargs = client.fine_tuning.jobs.create.call_args[1]
        self.assertEqual(call_kwargs["hyperparameters"]["n_epochs"], 3)
        self.assertEqual(call_kwargs["suffix"], "test")


class TestCheckStatus(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_succeeded(self, mock_get):
        client = MagicMock()
        client.fine_tuning.jobs.retrieve.return_value = _mock_job("succeeded")
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        status = ft.check_status("ftjob-test123")
        self.assertEqual(status.status, "succeeded")
        self.assertIsNotNone(status.fine_tuned_model)
        self.assertEqual(status.trained_tokens, 5000)

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_failed(self, mock_get):
        client = MagicMock()
        client.fine_tuning.jobs.retrieve.return_value = _mock_job("failed")
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        status = ft.check_status("ftjob-test123")
        self.assertEqual(status.status, "failed")
        self.assertIsNotNone(status.error_message)


class TestWaitForCompletion(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.time.sleep")
    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner.check_status")
    def test_polls_until_success(self, mock_check, mock_sleep):
        statuses = [
            APIFineTuneStatus(job_id="j1", status="queued", file_id="f1"),
            APIFineTuneStatus(job_id="j1", status="running", file_id="f1"),
            APIFineTuneStatus(
                job_id="j1", status="succeeded", file_id="f1",
                fine_tuned_model="ft:model",
            ),
        ]
        mock_check.side_effect = statuses

        ft = APIFineTuner(api_key="sk-test")
        result = ft.wait_for_completion("j1", poll_interval=1)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(mock_check.call_count, 3)

    @patch("socialscikit.quantikit.api_finetuner.time.sleep")
    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner.check_status")
    def test_progress_callback(self, mock_check, mock_sleep):
        mock_check.return_value = APIFineTuneStatus(
            job_id="j1", status="succeeded", file_id="f1",
        )
        messages = []
        ft = APIFineTuner(api_key="sk-test")
        ft.wait_for_completion("j1", progress_callback=messages.append)
        self.assertEqual(len(messages), 1)
        self.assertIn("succeeded", messages[0])


class TestCancelJob(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_cancel(self, mock_get):
        client = MagicMock()
        client.fine_tuning.jobs.retrieve.return_value = _mock_job("cancelled")
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        status = ft.cancel_job("ftjob-test123")
        client.fine_tuning.jobs.cancel.assert_called_once_with("ftjob-test123")
        self.assertEqual(status.status, "cancelled")


class TestPredict(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_predict(self, mock_get):
        client = MagicMock()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "positive"
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        ft._system_prompt = "classify"
        preds = ft.predict(["hello", "world"], "ft:model")
        self.assertEqual(preds, ["positive", "positive"])
        self.assertEqual(client.chat.completions.create.call_count, 2)


class TestFormatStatus(unittest.TestCase):

    def test_format_succeeded(self):
        s = APIFineTuneStatus(
            job_id="j1", status="succeeded", file_id="f1",
            fine_tuned_model="ft:model", trained_tokens=5000,
        )
        text = APIFineTuner.format_status(s)
        self.assertIn("j1", text)
        self.assertIn("训练完成", text)
        self.assertIn("ft:model", text)
        self.assertIn("5,000", text)

    def test_format_failed(self):
        s = APIFineTuneStatus(
            job_id="j1", status="failed", file_id="f1",
            error_message="bad data",
        )
        text = APIFineTuner.format_status(s)
        self.assertIn("训练失败", text)
        self.assertIn("bad data", text)


class TestListEvents(unittest.TestCase):

    @patch("socialscikit.quantikit.api_finetuner.APIFineTuner._get_client")
    def test_list(self, mock_get):
        client = MagicMock()
        evt = MagicMock()
        evt.message = "Step 100: loss=0.5"
        evt.created_at = 1700000100
        evt.level = "info"
        events_resp = MagicMock()
        events_resp.data = [evt]
        client.fine_tuning.jobs.list_events.return_value = events_resp
        mock_get.return_value = client

        ft = APIFineTuner(api_key="sk-test")
        events = ft.list_events("j1")
        self.assertEqual(len(events), 1)
        self.assertIn("loss", events[0]["message"])


if __name__ == "__main__":
    unittest.main()

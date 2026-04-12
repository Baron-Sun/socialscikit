"""OpenAI API fine-tuning pipeline.

Upload labeled data as JSONL → submit a fine-tuning job → poll status →
predict with the fine-tuned model.  Runs entirely on OpenAI's servers.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported base models for fine-tuning
# ---------------------------------------------------------------------------

FINETUNE_MODELS = [
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "gpt-3.5-turbo-0125",
]

MIN_EXAMPLES = 10  # OpenAI minimum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class APIFineTuneConfig:
    """Configuration for an API fine-tuning job."""

    model: str = "gpt-4o-mini-2024-07-18"
    n_epochs: str = "auto"  # "auto" or integer string
    batch_size: str = "auto"
    learning_rate_multiplier: str = "auto"
    suffix: str = ""
    seed: int = 42


@dataclass
class APIFineTuneStatus:
    """Snapshot of a fine-tuning job's state."""

    job_id: str = ""
    status: str = ""  # validating_files | queued | running | succeeded | failed | cancelled
    file_id: str = ""
    fine_tuned_model: str | None = None
    trained_tokens: int | None = None
    error_message: str | None = None
    created_at: int = 0
    finished_at: int | None = None


@dataclass
class APIFineTuneResult:
    """Final result after training + prediction."""

    fine_tuned_model: str = ""
    job_id: str = ""
    status: APIFineTuneStatus = field(default_factory=APIFineTuneStatus)
    predictions: list[str] = field(default_factory=list)
    probabilities: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# APIFineTuner
# ---------------------------------------------------------------------------


class APIFineTuner:
    """End-to-end OpenAI API fine-tuning for text classification.

    Usage::

        ft = APIFineTuner(api_key="sk-...")
        path = ft.prepare_jsonl(df, "text", "label", ["pos", "neg"])
        file_id = ft.upload_file(path)
        job_id = ft.create_job(file_id)
        status = ft.wait_for_completion(job_id)
        preds = ft.predict(["new text"], status.fine_tuned_model)
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None
        self._system_prompt: str = ""

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # JSONL preparation
    # ------------------------------------------------------------------

    def prepare_jsonl(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        labels: list[str],
        label_definitions: dict[str, str] | None = None,
        seed: int = 42,
    ) -> str:
        """Convert labeled DataFrame to OpenAI fine-tuning JSONL format.

        Returns the path to the temporary JSONL file.
        """
        sub = df[[text_col, label_col]].dropna()
        if len(sub) < MIN_EXAMPLES:
            raise ValueError(
                f"至少需要 {MIN_EXAMPLES} 条标注数据，当前仅 {len(sub)} 条。"
            )

        unknown = set(sub[label_col].unique()) - set(labels)
        if unknown:
            raise ValueError(f"发现未定义的标签：{unknown}。请检查数据。")

        # Build system prompt
        system = f"将文本分类为以下类别之一：{', '.join(labels)}。"
        if label_definitions:
            defs = "\n".join(f"- {k}: {v}" for k, v in label_definitions.items())
            system += f"\n\n类别定义：\n{defs}"
        system += "\n\n只输出类别名称，不要输出其他内容。"
        self._system_prompt = system

        # Shuffle
        sub = sub.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Write JSONL
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for _, row in sub.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(row[text_col])},
                    {"role": "assistant", "content": str(row[label_col])},
                ]
            }
            tmp.write(json.dumps(example, ensure_ascii=False) + "\n")
        tmp.close()
        logger.info("Prepared %d training examples → %s", len(sub), tmp.name)
        return tmp.name

    # ------------------------------------------------------------------
    # File upload
    # ------------------------------------------------------------------

    def upload_file(self, jsonl_path: str) -> str:
        """Upload JSONL file to OpenAI. Returns the file ID."""
        client = self._get_client()
        with open(jsonl_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        logger.info("Uploaded file: %s", response.id)
        return response.id

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(
        self,
        file_id: str,
        config: APIFineTuneConfig | None = None,
    ) -> str:
        """Create a fine-tuning job. Returns the job ID."""
        config = config or APIFineTuneConfig()
        client = self._get_client()

        hyperparams: dict[str, Any] = {}
        if config.n_epochs != "auto":
            hyperparams["n_epochs"] = int(config.n_epochs)
        if config.batch_size != "auto":
            hyperparams["batch_size"] = int(config.batch_size)
        if config.learning_rate_multiplier != "auto":
            hyperparams["learning_rate_multiplier"] = float(
                config.learning_rate_multiplier
            )

        kwargs: dict[str, Any] = {
            "training_file": file_id,
            "model": config.model,
            "seed": config.seed,
        }
        if hyperparams:
            kwargs["hyperparameters"] = hyperparams
        if config.suffix:
            kwargs["suffix"] = config.suffix

        job = client.fine_tuning.jobs.create(**kwargs)
        logger.info("Created fine-tuning job: %s", job.id)
        return job.id

    def check_status(self, job_id: str) -> APIFineTuneStatus:
        """Retrieve the current status of a fine-tuning job."""
        client = self._get_client()
        job = client.fine_tuning.jobs.retrieve(job_id)
        error_msg = None
        if job.error and job.error.message:
            error_msg = job.error.message
        return APIFineTuneStatus(
            job_id=job.id,
            status=job.status,
            file_id=job.training_file,
            fine_tuned_model=job.fine_tuned_model,
            trained_tokens=job.trained_tokens,
            error_message=error_msg,
            created_at=job.created_at,
            finished_at=job.finished_at,
        )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 7200,
        progress_callback: Callable[[str], None] | None = None,
    ) -> APIFineTuneStatus:
        """Poll until the job finishes or times out."""
        start = time.time()
        while True:
            status = self.check_status(job_id)
            if progress_callback:
                progress_callback(
                    f"[{status.status}] Job {job_id}"
                    + (f" | tokens: {status.trained_tokens}" if status.trained_tokens else "")
                )
            if status.status in ("succeeded", "failed", "cancelled"):
                return status
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"训练超时（超过 {timeout // 60} 分钟）。"
                    f"请使用任务 ID 手动检查：{job_id}"
                )
            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> APIFineTuneStatus:
        """Cancel a running fine-tuning job."""
        client = self._get_client()
        client.fine_tuning.jobs.cancel(job_id)
        return self.check_status(job_id)

    def list_events(self, job_id: str, limit: int = 20) -> list[dict]:
        """List recent events for a fine-tuning job."""
        client = self._get_client()
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=limit
        )
        return [
            {"message": e.message, "created_at": e.created_at, "level": e.level}
            for e in events.data
        ]

    # ------------------------------------------------------------------
    # Prediction with fine-tuned model
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: list[str],
        fine_tuned_model: str,
        system_prompt: str | None = None,
    ) -> list[str]:
        """Classify texts using the fine-tuned model."""
        client = self._get_client()
        sys_prompt = system_prompt or self._system_prompt

        predictions = []
        for text in texts:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ]
            resp = client.chat.completions.create(
                model=fine_tuned_model,
                messages=messages,
                temperature=0,
                max_tokens=50,
            )
            label = resp.choices[0].message.content.strip()
            predictions.append(label)
        return predictions

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_status(status: APIFineTuneStatus) -> str:
        """Format a status object as a human-readable string."""
        status_map = {
            "validating_files": "验证文件中",
            "queued": "排队中",
            "running": "训练中",
            "succeeded": "训练完成",
            "failed": "训练失败",
            "cancelled": "已取消",
        }
        lines = [
            f"任务 ID: {status.job_id}",
            f"状态: {status_map.get(status.status, status.status)}",
        ]
        if status.fine_tuned_model:
            lines.append(f"模型 ID: {status.fine_tuned_model}")
        if status.trained_tokens:
            lines.append(f"训练 tokens: {status.trained_tokens:,}")
        if status.error_message:
            lines.append(f"错误: {status.error_message}")
        return "\n".join(lines)

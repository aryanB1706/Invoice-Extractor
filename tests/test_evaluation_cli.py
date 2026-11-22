"""CLI tests for the evaluation harness."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from invoiceiq.config import Settings
from invoiceiq.evaluation import run_eval, runner
from invoiceiq.schemas import (
    DocumentQuality,
    ExtractionResponse,
    Invoice,
    TokenUsage,
    ValidationReport,
)


def _invoice_payload() -> dict[str, Any]:
    return {
        "is_invoice": True,
        "vendor": "Acme Supplies Pvt Ltd",
        "invoice_number": "INV-2041",
        "invoice_date": "2026-03-14",
        "currency": "INR",
        "line_items": [
            {"description": "A4 paper", "quantity": 10, "amount": 3200.0}
        ],
        "tax_total": 576.0,
        "total": 3776.0,
    }


def _write_dataset(directory: Path, payloads: dict[str, dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stem, payload in payloads.items():
        (directory / f"{stem}.pdf").write_bytes(b"%PDF-1.4 placeholder")
        (directory / f"{stem}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _configured_settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-key"),
        groq_model="fake-model",
    )


def test_cli_runs_with_injected_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    payloads = {"L01": _invoice_payload()}
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    _write_dataset(data_dir, payloads)

    def fake_process(
        path: Path,
        *,
        on_pages_loaded: Callable[[DocumentQuality], None] | None = None,
    ) -> ExtractionResponse:
        if on_pages_loaded is not None:
            on_pages_loaded(
                DocumentQuality(
                    page_count=1,
                    ocr_pages=0,
                    text_characters=100,
                )
            )
        return ExtractionResponse(
            invoice=Invoice.model_validate(payloads[path.stem]),
            validation=ValidationReport(is_valid=True),
            needs_review=False,
            model="fake-model",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(runner, "process_document", fake_process)
    monkeypatch.setattr(run_eval, "get_settings", _configured_settings)

    exit_code = run_eval.main(
        ["--data-dir", str(data_dir), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Overall accuracy" in captured.out
    assert (output_dir / "evaluation.json").is_file()
    assert (output_dir / "report.md").is_file()


def test_cli_rejects_unconfigured_groq(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_eval, "get_settings", lambda: Settings(_env_file=None)
    )

    exit_code = run_eval.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Groq is not configured" in captured.err


def test_cli_rejects_negative_tolerance(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_eval, "get_settings", _configured_settings)

    exit_code = run_eval.main(["--tolerance", "-1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "tolerance" in captured.err

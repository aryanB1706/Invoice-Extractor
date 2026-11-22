# InvoiceIQ

AI invoice extraction pipeline built with Python, LangChain, Groq, FastAPI,
PyMuPDF, Tesseract OCR, Pydantic, Pandas, and Pytest.

The application processes uploaded invoices through document processing, LLM
extraction, schema validation, and mathematical business validation. Dataset
evaluation is available for the labelled assignment sample pack.

## Architecture

```text
Document -> Document Processing -> LLM Extraction -> Validation -> API / CLI
                                         |
                                     Evaluation
```

Each stage has a single responsibility and communicates through the Pydantic
models in `src/invoiceiq/schemas.py`.

## Extraction output

`extract_invoice(document_text)` returns exactly this structure:

```json
{
  "is_invoice": true,
  "vendor": "Acme Supplies Pvt Ltd",
  "invoice_number": "INV-2041",
  "invoice_date": "2026-03-14",
  "currency": "INR",
  "line_items": [
    {
      "description": "A4 paper, 500 sheets",
      "quantity": 10,
      "amount": 3200.0
    }
  ],
  "tax_total": 576.0,
  "total": 3776.0
}
```

For non-invoice documents, `is_invoice` is `false` and every other field is
`null`, including `line_items`. Token usage and model metadata are available
separately through `InvoiceExtractor.extract()`.

### Validation layers

- **Schema validation** (`schemas.py`): strict field types, required keys,
  `YYYY-MM-DD` dates, finite numbers, no extra keys, and null-only non-invoices.
- **Mathematical validation** (`validation/validator.py`): required business
  fields and line-item/tax totals. This is intentionally separate from schema
  validation and runs in the next stage.

```python
from invoiceiq.extraction import extract_invoice

invoice = extract_invoice(document_text)
payload = invoice.model_dump(mode="json")
```

## Project structure

```text
InvoiceIQ/
├── .env.example                 # Environment variable template
├── .gitignore                   # Excludes secrets, caches, and local data
├── pyproject.toml               # Package metadata, dependencies, tool config
├── requirements.txt             # Runtime and development dependencies
├── README.md                    # Project overview and setup
├── data/
│   ├── samples/                 # Input invoices for manual testing
│   └── expected/                # Ground-truth JSON for evaluation
├── src/invoiceiq/
│   ├── __init__.py              # Package version and CLI entry point
│   ├── config.py                # Environment-based settings (API key, model, OCR)
│   ├── schemas.py               # Shared Pydantic models
│   ├── cli.py                   # Local command-line smoke test
│   ├── document_processing/     # PDF text extraction and Tesseract OCR
│   │   ├── loader.py            # Detects file type and loads page text
│   │   └── ocr.py               # Image-to-text conversion
│   ├── extraction/              # LangChain + Groq extraction layer
│   │   ├── errors.py            # Domain errors for API, parsing, and config failures
│   │   ├── prompts.py           # System and human prompt templates
│   │   └── extractor.py         # Groq client factory and extractor
│   ├── validation/              # Rule-based checks on extracted invoices
│   │   └── validator.py         # Required fields, totals, and line-item checks
│   ├── api/                     # FastAPI application
│   │   ├── app.py               # App factory and health endpoint
│   │   └── routes.py            # Upload/extract endpoint
│   └── evaluation/              # Accuracy measurement
│       ├── metrics.py           # Per-field comparison
│       └── runner.py            # Dataset-level aggregation
└── tests/                       # Pytest suite for config, schemas, API, and loading
```

### Folder purposes

| Path | Purpose |
| --- | --- |
| `src/invoiceiq/` | Importable application package. |
| `src/invoiceiq/document_processing/` | Converts PDFs and images into text. |
| `src/invoiceiq/extraction/` | Owns prompts and all LLM/Groq access. |
| `src/invoiceiq/validation/` | Applies deterministic business rules to extracted data. |
| `src/invoiceiq/api/` | Exposes the pipeline over HTTP. |
| `src/invoiceiq/evaluation/` | Scores predictions against labelled invoices. |
| `data/samples/` | Local invoices used for development. Ignored by Git. |
| `data/expected/` | Labelled results used by evaluation. Ignored by Git. |
| `tests/` | Automated checks for the skeleton and future stages. |

### Important files

| File | Purpose |
| --- | --- |
| `src/invoiceiq/config.py` | Single source of truth for environment configuration. |
| `src/invoiceiq/schemas.py` | Shared `Invoice`, `LineItem`, `PageText`, and validation models. |
| `src/invoiceiq/extraction/extractor.py` | Reads the Groq key and model from settings; no hardcoded credentials. |
| `src/invoiceiq/api/app.py` | Creates the FastAPI app and registers routes. |
| `.env` | Local secrets. Never committed. |
| `.env.example` | Safe, commit-friendly template for `.env`. |

## Setup

### 1. Create a virtual environment and install dependencies

With `uv`:

```powershell
uv sync --extra dev
```

Or with `pip`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Then set `GROQ_API_KEY` in `.env`. `GROQ_MODEL` is already provided and can be
changed without modifying code. Use a Groq text model that supports tool calling
and structured output; `llama-3.3-70b-versatile` is the default recommendation.
If Tesseract is not on `PATH`, set `TESSERACT_CMD` to the full executable path.

Generation settings are configurable with `GROQ_TEMPERATURE`,
`GROQ_MAX_TOKENS`, `GROQ_TIMEOUT_SECONDS`, and `GROQ_MAX_RETRIES`.

### 3. Install Tesseract OCR

- Windows: install from the UB Mannheim Tesseract build and set `TESSERACT_CMD`.
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

### 4. Run the API

```powershell
uv run uvicorn invoiceiq.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

`POST /extract` accepts a PDF, JPG, JPEG, PNG, TIFF, or BMP file and returns the
invoice, validation report, and a `needs_review` flag. The versioned alias
`POST /api/v1/extract` is also available. `GET /health` checks API availability.

### 5. Run the CLI

```powershell
uv run invoiceiq data/samples/example.pdf
```

### 6. Run tests and linting

```powershell
uv run pytest
uv run ruff check .
```

The default suite is offline: document processing, OCR, LLM calls, and the API
are exercised with local fakes, so no `GROQ_API_KEY` or network is required.

Optional checks over the downloaded assignment pack:

```powershell
# Requires data/samples/assignment with the 20 documents
$env:INVOICEIQ_DATASET_TESTS = "1"
uv run pytest -m dataset

# Real Groq call; checks wiring only, not accuracy
$env:INVOICEIQ_LIVE_TESTS = "1"
uv run pytest -m live
```

Dataset and live tests are skipped by default. The assignment pack lives under
`data/samples/assignment/` and is ignored by Git. No accuracy numbers are
asserted in the test suite; measure accuracy separately on the labelled set.

### 7. Evaluate the labelled assignment set

With `GROQ_API_KEY` and `GROQ_MODEL` configured, run this from the project root:

```powershell
python -m evaluation.run_eval
```

The command discovers every PDF/image in
`data/samples/assignment/labelled/` and pairs it with the same-stem JSON
ground-truth file (for example, `L01.pdf` and `L01.json`). It runs the normal
document-processing, extraction, and validation pipeline on each document.
Missing or unmatched files stop the evaluation rather than being skipped.
Optional `--data-dir`, `--output-dir`, and `--tolerance` arguments override the
defaults. The installed `invoiceiq-eval` command is also available.

Results are written to `data/outputs/evaluation/` by default:

- `predictions.json` and `predictions/<document>.json` contain each actual
  prediction, model/usage metadata, and any pipeline error.
- `evaluation.json` contains field comparisons, per-document details, and
  aggregate metrics.
- `report.md` contains the readable results, matching rules, and tuning
  disclosure.

Matching is intentionally strict and is also recorded in the generated report:

- Text is compared after Unicode NFKC normalization, whitespace collapsing,
  and case folding; punctuation and digits are preserved.
- Dates must parse as ISO `YYYY-MM-DD` dates and match exactly.
- Numeric fields use an absolute tolerance of `0.01` by default, so `1200` and
  `1200.00` match but meaningful numeric differences do not.
- `is_invoice` uses exact boolean matching. Null only matches null.
- Line items are matched in printed order. The item count and each item's
  normalized description, quantity, and amount must all match; extra, missing,
  or misordered items are not ignored.
- Overall accuracy is the unweighted mean of the eight field accuracies.
  Failed extractions count as incorrect in every field and remain in the
  denominator.

The default `GROQ_MAX_TOKENS` was raised globally from 2048 to 8192; it is not
configured per document. No model or prompt tuning is performed by evaluation.

## Status

Implemented:

- Configuration management
- Shared Pydantic schemas
- Digital PDF text extraction with PyMuPDF
- Per-page OCR fallback for scanned or low-text PDF pages
- Automatic rotation recovery for upside-down or rotated scans
- JPG, JPEG, PNG, TIFF, and BMP OCR support
- Graceful handling of blank pages, unreadable images, and corrupt PDFs
- LangChain `ChatGroq` client configured entirely from environment variables
- Strict invoice prompt with line-item, date, and non-invoice rules
- Reusable `extract_invoice(document_text)` function returning the exact JSON schema
- Strict Pydantic schema validation for required keys, types, dates, and non-invoices
- Structured invoice extraction with token-usage capture
- Domain errors for rate limits, timeouts, authentication, API failures, and parsing
- Validation for required fields and line-item/tax totals
- FastAPI health endpoint
- PDF and image upload extraction endpoint with upload checks and HTTP errors
- Mathematical validation with configurable tolerance and review flags
- CLI document loader
- In-memory upload processing without temporary files
- Evaluation harness for the labelled assignment set with field-level,
  line-item, classification, and overall metrics
- End-to-end integration tests for digital PDFs, scans, images, multi-page
  documents, non-invoices, missing fields, unreadable files, corrupt PDFs, and
  Groq failures
- Optional dataset and live-API test markers

Reserved for the next prompts:

- Upload persistence

# InvoiceIQ

AI invoice extraction pipeline built with Python, LangChain, Groq, FastAPI,
PyMuPDF, Tesseract OCR, Pydantic, Pandas, and Pytest.

This repository currently contains the project skeleton. The pipeline is wired
into clear stages, but extraction and dataset evaluation are reserved for the
next implementation prompts.

## Architecture

```text
Document -> Document Processing -> LLM Extraction -> Validation -> API / CLI
                                         |
                                     Evaluation
```

Each stage has a single responsibility and communicates through the Pydantic
models in `src/invoiceiq/schemas.py`.

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
│   │   ├── prompts.py           # System and human prompt templates
│   │   └── extractor.py         # Groq client factory and extractor
│   ├── validation/              # Rule-based checks on extracted invoices
│   │   └── validator.py         # Required fields, totals, and date checks
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
changed without modifying code. If Tesseract is not on `PATH`, set
`TESSERACT_CMD` to the full executable path.

### 3. Install Tesseract OCR

- Windows: install from the UB Mannheim Tesseract build and set `TESSERACT_CMD`.
- macOS: `brew install tesseract`
- Linux: `sudo apt install tesseract-ocr`

### 4. Run the API

```powershell
uv run uvicorn invoiceiq.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

### 5. Run the CLI

```powershell
uv run invoiceiq data/samples/example.pdf
```

### 6. Run tests and linting

```powershell
uv run pytest
uv run ruff check .
```

## Status

Implemented:

- Configuration management
- Shared Pydantic schemas
- Digital PDF text extraction with PyMuPDF
- Per-page OCR fallback for scanned or low-text PDF pages
- JPG, JPEG, PNG, TIFF, and BMP OCR support
- Graceful handling of blank pages, unreadable images, and corrupt PDFs
- Basic validation rules
- FastAPI health endpoint
- CLI document loader
- Tests for digital PDFs, scanned PDFs, images, and failure cases

Reserved for the next prompts:

- Groq prompt invocation and structured extraction
- Upload persistence and full `/extract` behavior
- Dataset evaluation metrics
- Accuracy reporting

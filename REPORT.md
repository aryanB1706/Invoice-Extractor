# InvoiceIQ Final Report

- Generated: 2026-11-22T12:45:45+00:00
- Model: `openai/gpt-oss-120b`
- Documents processed: 20 (20 successful, 0 failed)

## Project overview

InvoiceIQ accepts PDF and image invoices, extracts structured invoice fields, validates required values and arithmetic, and exposes the pipeline through a FastAPI endpoint and command-line tools.

## Architecture and workflow

1. PyMuPDF reads embedded PDF text; Tesseract OCR handles images and scanned or low-text PDF pages.
2. A LangChain `ChatGroq` structured-output call maps document text to the Pydantic `Invoice` schema.
3. Schema checks reject invalid dates, non-finite numbers, extra fields, and populated fields on non-invoices.
4. Business validation checks required invoice fields, line-item completeness, and whether line amounts plus tax equal the printed total.
5. The batch command writes one JSON file per source document; labelled predictions are compared to Ground Truth from that same run.

## Model selection

`openai/gpt-oss-120b` is the configured Groq model for this run. Groq's model documentation lists tool use, JSON Schema mode, and reasoning capabilities, which fit the structured invoice response. No comparative model benchmark was run, so model selection is configuration-driven rather than a claim that this model outperforms alternatives.

Source: https://console.groq.com/docs/models (checked 2026-11-22).

## Extraction approach and prompting

The extractor uses a strict Pydantic schema with nullable fields. The system prompt directs the model to use printed evidence, preserve every line-item row, map values by table headers, distinguish quantity/unit price/line amount/tax/grand total, normalize dates, avoid unsupported calculations, and return null for missing or ambiguous values. Unambiguous currency symbols may map to a currency code; address-based currency guessing is disallowed.

### Approach that failed

An earlier long-invoice request using Groq function-calling structured output returned malformed tool-call JSON for L04 once. A later full run succeeded on that document. This was an observed output-format failure, not evidence that extraction always fails for long invoices. The global completion-token setting was raised to 8192; failed documents are now recorded explicitly rather than receiving fabricated values.

## Field-level accuracy

Accuracy is measured on the 10 labelled documents using the batch predictions in this submission.

| Field | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| is_invoice | 10 | 10 | 100.0% |
| vendor | 10 | 10 | 100.0% |
| invoice_number | 10 | 10 | 100.0% |
| invoice_date | 10 | 10 | 100.0% |
| currency | 6 | 10 | 60.0% |
| line_items | 7 | 10 | 70.0% |
| tax_total | 6 | 10 | 60.0% |
| total | 9 | 10 | 90.0% |

**Macro overall accuracy:** 85.0% (unweighted mean of the eight field accuracies).

Fully correct documents: 5/10. This is a small labelled set; the measured score is descriptive, not a population guarantee.

### Matching rules

- Text (vendor, invoice_number, currency): Unicode NFKC normalization, whitespace collapsing, and case folding. Punctuation and digits are kept.
- Dates (invoice_date): parsed as ISO YYYY-MM-DD and compared as dates.
- Numbers (tax_total, total, quantity, amount): compared with an absolute tolerance of 0.01. 1200 and 1200.00 are equivalent.
- is_invoice: exact boolean comparison.
- line_items: compared in printed order. The field matches only when the item count matches and every item matches on description, quantity, and amount. Extra or missing items are errors, not ignored.
- Null handling: null matches only null. A null prediction never matches a populated ground-truth value.
- Overall accuracy: unweighted mean of the eight field-level accuracies. Each field has equal weight; line_items is a strict whole-list match.
- Failed extractions count as incorrect for every field and are reported separately. They are not excluded from the denominator.

Line-item detail: 50/66 aligned items fully match; description 90.9%, quantity 75.8%, amount 89.4%, item-count 88.9%.

## Failure analysis

5/10 labelled documents contain at least one field mismatch. Categories may overlap.

| Category | Documents | Examples |
| --- | ---: | --- |
| OCR-associated errors | 3 | L02.jpg, L09.pdf, L10.pdf |
| Poor document quality | 0 | None observed |
| Incorrect vendor extraction | 0 | None observed |
| Invoice number extraction errors | 0 | None observed |
| Date extraction errors | 0 | None observed |
| Line-item extraction errors | 3 | L02.jpg, L09.pdf, L10.pdf |
| Tax and total mismatches | 4 | L02.jpg, L03.pdf, L09.pdf, L10.pdf |
| Non-invoice classification errors | 0 | None observed |
| LLM hallucinations / unsupported values | 0 | None observed |
| Missing information | 5 | L02.jpg, L03.pdf, L07.pdf, L09.pdf, L10.pdf |

### Human-review effectiveness

A document counts as having an extraction error when any evaluated field differs from Ground Truth. This is an empirical comparison on the small labelled set, not a calibrated confidence measure.

| Review status | Documents | With errors | Error rate | Mean wrong fields/document |
| --- | ---: | ---: | ---: | ---: |
| Flagged | 6 | 5 | 83.3% | 2.00 |
| Not flagged | 4 | 0 | 0.0% | 0.00 |

Flagged documents without a Ground Truth mismatch: 1.

### Line-item extraction errors

**L02.jpg** — OCR was used and item-level fields disagree with Ground Truth. OCR damage and downstream table interpretation are both plausible; this comparison cannot isolate the cause.
- Expected/predicted item count: 6/6.
- Matching descriptions/quantities/amounts: 6/6, 0/6, 6/6.
- item 1: quantity 150.0 != None
- item 2: quantity 20.0 != None
- item 3: quantity 10.0 != None

**L09.pdf** — OCR was used and item-level fields disagree with Ground Truth. OCR damage and downstream table interpretation are both plausible; this comparison cannot isolate the cause.
- Expected/predicted item count: 5/2.
- Matching descriptions/quantities/amounts: 0/5, 0/5, 0/5.
- item 1: description 'King fish (seer), cleaned, kg' != 'King fish (seer), cleaned, ig'; quantity 12.5 != None; amount 11500.0 != None
- item 2: description 'Tiger prawns 16/20, kg' != 'Tiger prawns 16/20, ee'; quantity 18.0 != None; amount 14040.0 != None
- item 3: missing on predicted side

**L10.pdf** — OCR was used and item-level fields disagree with Ground Truth. OCR damage and downstream table interpretation are both plausible; this comparison cannot isolate the cause.
- Expected/predicted item count: 5/5.
- Matching descriptions/quantities/amounts: 4/5, 0/5, 3/5.
- item 1: quantity 8.0 != None
- item 2: quantity 12.0 != None; amount 5040.0 != 380.0
- item 3: quantity 15.0 != None


### Tax and total mismatches

**L02.jpg** — Validation detected incomplete or inconsistent totals: Cannot verify total because the tax amount is missing.
- tax_total: expected 3529.8; predicted null.
- Validation: Cannot verify total because the tax amount is missing.

**L03.pdf** — Validation detected incomplete or inconsistent totals: Cannot verify total because the tax amount is missing.
- tax_total: expected 9788.4; predicted null.
- Validation: Cannot verify total because the tax amount is missing.

**L09.pdf** — Validation detected incomplete or inconsistent totals: Total is missing.; Cannot verify total because the tax amount is missing.
- tax_total: expected 2032.5; predicted null.
- total: expected 42682.5; predicted null.
- Validation: Total is missing.
- Validation: Cannot verify total because the tax amount is missing.


### Missing information

**L02.jpg** — Ground Truth contains values absent from the prediction, and OCR was used. OCR/table recovery is a plausible contributor, but downstream field selection cannot be ruled out.
- currency is null; Ground Truth is INR.
- tax_total is null; Ground Truth is 3529.8.
- line_items[1].quantity is missing.
- line_items[2].quantity is missing.
- line_items[3].quantity is missing.

**L03.pdf** — Ground Truth contains values absent from the prediction. The document was read without OCR, so the mismatch is in extraction/field selection; the exact model-side cause is not established.
- tax_total is null; Ground Truth is 9788.4.

**L07.pdf** — Ground Truth contains values absent from the prediction. The document was read without OCR, so the mismatch is in extraction/field selection; the exact model-side cause is not established.
- currency is null; Ground Truth is USD.

### Highest-priority improvements

1. Improve OCR/table row and column recovery, especially quantities and item boundaries; route OCR-derived invoices with missing line-item fields to human review.
2. Strengthen tax-versus-grand-total separation and arithmetic checks. Keep tax missing/field mix-ups visible instead of filling them by inference.
3. Retain the deterministic review gates: flagged documents currently have more mismatched fields on average than unflagged documents.
4. Use a separate development set for any prompt/model changes; preserve the labelled assignment set as a held-out evaluation.

## Cost analysis

- Pricing model: `openai/gpt-oss-120b`
- Source: https://console.groq.com/docs/model/openai/gpt-oss-120b (checked 2026-11-22)
- Standard input: $0.150/1M tokens; cached input: $0.075/1M; output: $0.600/1M.
- Input tokens observed: 22,619; output tokens observed: 15,430; cached input tokens observed: 7,936.
- Documents with complete input/output usage: 20/20.
- Cost for documents with complete usage: $0.01205565.
- Total API list-price estimate: $0.01205565.
- Average per successfully processed document: $0.00060278.

Per request: (uncached input tokens × input USD/1M + cached input tokens × cached-input USD/1M + output tokens × output USD/1M) / 1,000,000. A cached-input discount is applied only when the provider reports cached token counts. Missing input/output metadata is never treated as zero; if any request is missing either count, total API cost is reported as unknown.

The batch used a Groq free-tier API key. The listed-price estimate is provided for comparison; the inference API does not expose account billing, so the actual charge is not independently verified. Usage within the free quota is expected to cost $0.

Missing token metadata is listed in `outputs/cost_analysis.json`; missing counts are not treated as zero. Invalid cached-token counts are listed separately and leave total cost unknown.

## Outputs and Ground Truth handling

The batch wrote 20 JSON files to `outputs/`, named from the source stems. Each record stores the original filename, split, prediction, validation, review flag, token usage, and any error. Unlabelled records have no Ground Truth field or fabricated labels. Evaluation uses only the 10 actual same-stem labelled JSON files.

## Known limitations

- Only 10 documents have Ground Truth; unlabelled documents cannot be scored for accuracy.
- OCR can lose table structure or confuse small text; OCR use does not prove that OCR caused a specific mismatch.
- Structured extraction may omit quantities or tax fields even when text is available. Validation flags incomplete fields but does not invent or derive them.
- The batch used a Groq free-tier API key. The inference response does not provide an account billing statement, so the actual charge is unverified.
- Prompt/OCR changes were made after reviewing the labelled-set failure categories. This run is a post-analysis diagnostic, not an independent holdout measurement.

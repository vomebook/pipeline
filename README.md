# Small PDF render worker

This fork is reserved for rendering PDFs smaller than 100 MiB into Reader page
streams. `anftm/pipeline` continues to own large-PDF rendering and OCR. The
`Render Small PDF OCR Inputs` workflow is manual-only until the original small
render schedule has been disabled in `anftm/pipeline`. Running both copies of
that schedule would select the same pending books.

Before enabling this worker, configure `HF_TOKEN`, `PAGES_TOKEN`, and
`PAGES_REPO` repository secrets, and coordinate publication of the shared
Reader-Assets dataset and Pages sidecar with the original pipeline. Keep
other jobs in the original repository. See `PDF_OCR.md` for asset contracts.

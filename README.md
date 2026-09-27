# Small PDF render worker

This fork is reserved for rendering PDFs smaller than 100 MiB into Reader page
streams. `anftm/pipeline` continues to own large-PDF rendering and OCR. The
`Render Small PDF OCR Inputs` runs every 15 minutes. The original small render
workflow in `anftm/pipeline` must remain disabled; running both would select
the same pending books.

This fork publishes its completed page streams to the shared Reader-Assets
dataset. It then dispatches `anftm/pipeline`'s `Publish Reader Index` workflow,
which serializes Pages sidecar publication with the original pipeline jobs.
For nonempty render batches it also dispatches the main pipeline's PDF text
publication workflow, so native-text books receive complete search indexes
without running image recognition. Configure `HF_TOKEN` and a `PIPELINE_TOKEN`
with Actions write permission on `anftm/pipeline` as repository secrets. The
original pipeline still owns OCR.
See `PDF_OCR.md` for asset contracts.

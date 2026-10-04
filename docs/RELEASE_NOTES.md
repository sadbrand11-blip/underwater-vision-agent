# v0.5.3-public.1 — public delivery

A fresh, sanitized source history based on 0.5.3. English architecture/GIF/metrics
homepage and Chinese instructions; native remains the default runtime.

## Run without a key or weights

Install `requirements-demo.txt`, then `python run_demo.py`. Open `/showcase`.
Exposure assessment and bounded correction use actual local code. The scheduler
is scripted and marked as such. Detection without a checkpoint is unavailable,
never an empty result falsely claiming no objects.

## Optional detector asset

`sodd_detector.pt` is the inspected six-class default checkpoint (about 76 MB).
Its size, SHA256 and taxonomy are in `sodd_detector.release.json` and source
`models/release.json`. Install CPU detector dependencies, then use
`python run_demo.py --with-detector`; the launcher verifies the asset before use.
MIT license is provided as `LICENSE-model.txt`. Training data attribution and
example-image CC BY 4.0 notices are in `THIRD_PARTY_NOTICES.md`.

## Evidence and limitations

Historical formal, adaptive and retrieval scores retain separate version/scope
labels and known failures. The public TF-IDF corpus is independent of frozen C1.
This release does not claim improved visual accuracy or live LLM capability.
No paid API calls or model retraining were performed for public delivery.

The immutable initial source tag is retained. `main` includes the subsequent
optional SDK compatibility fix (Pydantic 2.13.5 for LangGraph + MCP), explicit
optional Embedding dependency errors, readable vertical architecture layout,
and anonymous publication verification. The
checkpoint bytes and visual algorithms are unchanged. Clone `main` for current
instructions; the Release continues to host the original verified checkpoint.

Raw datasets, personal memory, secrets, session logs, and other checkpoints are
not included in the public Git history. See `docs/PUBLIC_VERIFICATION.md` for
the verification counts, skips and the deliberately retained unreliable demo.

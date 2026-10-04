# Third-party data and dependencies

## SODD example imagery and training data

The example JPEG in `assets/examples/` and the original image pixels visible in
`assets/demo.gif` come from **SODD — Subaquatic Object Detection Dataset**:

- Hafiz Muhammad Ahmad Imam, Erlend Andreas Basso, Simon Andreas Hoff,
  Hergys Rexha, Sébastien Lafond, and Bogdan Iancu.
- Official record: <https://zenodo.org/records/10230328>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Exact source file and SHA256 are recorded in `assets/examples/provenance.json`.
- Changes: simulated brightness multiplication by 0.6; optional all-black stress
  input; resized presentation, bounding-box overlays and application UI.
- These are simulated image perturbations, not new camera acquisitions.

The default six-class checkpoint was trained using SODD. It is distributed by the
project author under MIT in a separate GitHub Release, with the same attribution.
It is an experimental detector, not a navigation or safety-certified model.

## Scope of MIT

`LICENSE` covers this project's source and the released project checkpoint.
It does not relicense SODD image pixels, third-party papers, or dependencies.
Research knowledge cards are short attributed summaries; full papers and datasets
are not bundled. Dependencies retain their own licenses, including PyTorch,
Torchvision, scikit-learn, OpenCV, Flask, Transformers, LangGraph and MCP.

TorchVision is used for the detector architecture and COCO initialization.
Its original BSD 3-Clause copyright and conditions are retained in
[`licenses/TORCHVISION_BSD3.txt`](licenses/TORCHVISION_BSD3.txt), copied from
the [official v0.17.0 license](https://github.com/pytorch/vision/blob/v0.17.0/LICENSE).
MIT covers this project's contributions; it does not remove upstream notices.

## Test fixtures

Key-shaped strings in redaction tests are deliberately fabricated fixtures.
They are not credentials and must remain in the tests to verify sanitization.

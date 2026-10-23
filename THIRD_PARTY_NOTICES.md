# Third-party sources and licenses

## Current general-image release (0.6.0)

The optional detector uses the official TorchVision Faster R-CNN MobileNetV3
320 COCO checkpoint. The complete original head and official sparse category
mapping are retained. TorchVision's BSD 3-Clause conditions are reproduced in
[licenses/TORCHVISION_BSD3.txt](licenses/TORCHVISION_BSD3.txt).
The checkpoint is fetched from the official PyTorch host with a pinned SHA256;
it is not relicensed as a project-trained MIT checkpoint.

The natural photo shown in the 0.6.0 recording is NASA's Eileen Collins image,
accessed through the installed scikit-image `astronaut` example:
[official source and public-domain statement](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut).
The input photo and recording source frames stay outside the repository.
The GIF shows actual local tools, deterministic offline scheduling, resizing and
box overlays. Dark (brightness x0.45) and black variants are simulated, with no
object ground truth or claim of detection accuracy. When this photo is absent,
the no-download demo uses an explicitly labeled procedural gradient fixture.

COCO and SICE data are not distributed here. Each COCO image retains its original
license; see [COCO terms](https://cocodataset.org/#termsofuse). SICE use follows
its authors' conditions. No reference image is treated as camera exposure truth.

## Archived underwater release

The retained historical JPEG in `assets/examples/` comes from SODD, by Hafiz
Muhammad Ahmad Imam, Erlend Andreas Basso, Simon Andreas Hoff, Hergys Rexha,
Sébastien Lafond and Bogdan Iancu.
[Official record](https://zenodo.org/records/10230328),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Its exact source and SHA256 are in `assets/examples/provenance.json`.
Historical demo variants multiplied brightness by 0.6 or replaced it with black.
The separately released six-class SODD checkpoint and old GIF belong to the
archived version; they are not the 0.6.0 default detector or current metrics.

## Project source and test fixtures

MIT covers this project's contributions, not third-party datasets, papers,
upstream checkpoints or dependencies. Knowledge cards are short attributed
summaries; paper originals are not bundled. PyTorch, TorchVision, scikit-learn,
OpenCV, Flask, Transformers, LangGraph and MCP retain their own licenses.
Key-shaped strings in redaction tests are deliberately fabricated test fixtures,
not credentials.

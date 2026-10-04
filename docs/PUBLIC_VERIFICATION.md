# Public delivery verification — 2026-10-04

This snapshot is based on source version **0.5.3** and preserves historical
experiment scopes. It creates a new source history rather than exposing the
private development Git directory. No paid model calls or training were performed.

## Checks completed locally

| Check | Actual result |
|---|---|
| Fresh Python 3.11 lightweight environment | 6 public-entry tests passed; Torch and LangGraph absent before optional installation |
| Final public bootstrap regression | 8 passed, including a cached checkpoint without Torch and explicit optional Embedding dependency failures |
| Full Python regression with existing optional LangGraph environment | 257 passed, 4 skipped |
| Actual MCP SDK / subprocess suite | 17 passed, 1 skipped |
| Fresh public CPU environment with both optional SDKs | 261 passed, 2 skipped; Torch 2.2.0+cpu, LangGraph 1.2.12, MCP 2.3.0, Pydantic 2.13.5 |
| Existing web component checks | 12 conversation +6 memory +4 LangGraph scenarios passed |
| Browser verification | Actual exposure, correction and pipe-analysis paths visible at `/showcase` |
| Credentials / path audit | Whitelist export and pattern scan; fabricated redaction fixtures explicitly allowlisted |

Full-regression skips: two optional MCP protocol cases in that environment, one
Windows symbolic-link permission case, and one test that reconstructs a private
historical Git commit deliberately absent from this fresh history. The separate
MCP environment executes both actual protocol cases. Its remaining skip is the
Windows symbolic-link case. The first sandboxed MCP attempt encountered a Windows
local-pipe permission error; the permitted rerun passed without changing the tool.

Two initial public-compatibility failures were fixed: a private-history corpus
reconstruction check now explicitly skips when the old commit is absent; public
corpus warnings and identities are separated from historical C1. Their focused
rerun passed, followed by the full regression above. Native completion and visual
reliability rules were not weakened.

## Demo provenance

The GIF replays real browser result frames at reading-friendly durations. It is
not a recording of cloud reasoning and does not represent real-time latency.
The source image is the licensed SODD file recorded in `assets/examples/provenance.json`.
The dark input is a simulated brightness ×0.6 perturbation.

The pipe-analysis preset actually executes quality assessment, bounded Gamma
candidate generation, original/candidate detection, candidate quality and evidence
comparison. The target scope is `pipe_type2` + `pipe`; **zero pipe candidates** in
this illustration are preserved, along with the actual **unreliable** conclusion.
No boxes are invented. Detecting nothing does not establish that no target exists.
Six tools are executed; cloud HTTP attempts are zero. This is an illustration,
not a new detector benchmark.

## Publication boundary

Code is MIT. Example-image pixels remain SODD CC BY 4.0 with source and change
notices. Only the inspected default SODD checkpoint is published separately as a
Release asset. It is absent from Git history; size/SHA256 verification precedes
launcher loading. The checkpoint was also inspected for personal paths/key-shaped
metadata and loaded with Torch's `weights_only=True` mode.

Raw datasets, session/evaluation logs, personal memory, secrets and other weights
are excluded. The actual lightweight dependency snapshot is
`requirements-demo-lock.txt`; optional CPU detection uses Torch 2.2.0 and
Torchvision 0.17.0. Cross-platform installs use `requirements-demo.txt`.

Public corpus content and hashes differ from historical C1. LF checkout rules
make the new corpus identity reproducible across platforms. Default TF-IDF is a
public delivery choice, not a new held-out retrieval acceptance result.

Checks reduce known disclosure and reproducibility risks; they are not a claim
of comprehensive external security or annotation review. Public clone and asset
download verification are recorded in the delivery summary after publication.

## Anonymous publication verification

Anonymous clone, new-data-root exposure assessment/correction, explicit unavailable
detection, anonymous Release download, checkpoint SHA256 verification and actual
CPU pipe analysis all passed. The real analysis again returned zero pipe
candidates and unreliable status, with six tool calls and no paid LLM requests.

The first GitHub CI lightweight job passed. Its combined optional-runtime job
failed during installation: the inherited LangGraph lock pinned Pydantic 2.10.6,
while MCP 2.3.0 requires at least 2.12.0. The public configuration now pins 2.13.5
and retains the core Requests version. The initial failure remains in Actions
history; compatibility updates live on `main`, preserving the initial source tag.

The fresh combined environment initially reported 259 passed, two skipped and
one failure: missing optional Transformers raised an unclassified import error.
The encoder now checks model availability before imports and reports missing
dependencies as `RagUnavailable`; TF-IDF does not require them. The focused
rerun passed 72 tests with two skips, followed by the full **261 passed, two
skipped** run in 172.73 seconds. The final skips are the Windows symbolic-link
permission case and the intentionally absent private historical commit.

Changing the homepage also changes the public corpus hash. Its consistency guard
correctly blocked new sessions until the release hash was regenerated. The
updated snapshot has 119 chunks and retains the guard; current measurements and
reliability requirements were not relaxed.

## Public CI and current snapshot

Both jobs passed on commit `1064be5f8c191f2c67f36e1ef958c85feb58cce4`:
[GitHub Actions run](https://github.com/sadbrand11-blip/underwater-vision-agent/actions/runs/37196933597).
The lightweight job installs no detector or optional SDKs; the regression job
installs both optional SDKs and runs the full Python and three web-component
suites. This verifies a fresh Linux setup independently of the local Windows
environment. The initial failed jobs are retained rather than replaced.

The anonymous clone was fast-forwarded to this commit and its eight public-entry
tests passed. GitHub renders the Mermaid diagram and loaded the actual 960 x1540
GIF. The machine-readable [delivery summary](PUBLIC_DELIVERY.json) records the
checkpoint, corpus identity and verification scope without raw session logs.

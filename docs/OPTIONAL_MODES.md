# Optional modes

Quick Start requires only `requirements-demo.txt`. All optional dependencies are
installed into your public environment, not the private project's environment.

## Cloud scheduling

Copy `.env.example` to `.env`, set a local API key, restart, then explicitly choose
cloud mode at `/agent`. This is optional and may be billed by the provider.
This release's recording and validation made **zero paid API requests**.

## Embedding / Hybrid

Install `transformers==4.32.1`, `tokenizers==0.13.2`, `safetensors==0.3.2` and the
optional Torch dependencies. The local model is `BAAI/bge-small-zh-v1.5`, revision
`7999e1d3359715c523056ef9478215996d62a620`. See the verified model source in
`docs/11_RAG_OPTIMIZATION.md`. Public assets belong under
`D:\CodexData\optical_agent\public\rag`, independent of historical frozen C0/C1.
Without the model or a valid index, the comparison UI reports unavailable.
The supported default is TF-IDF; historical Hybrid acceptance does not apply to
the sanitized public corpus. Do not run historical `publish_rag.py` against this
public configuration.

## LangGraph

Install `requirements-langgraph.txt`, restart, and select LangGraph at `/agent`.
It supports adaptive mode only; a session pins its runtime after its first turn.
Native remains the default. See `docs/13_LANGGRAPH_RUNTIME.md` for graph semantics.
No cloud tracing is enabled by this project.

## MCP

Install `requirements-mcp.txt`, then use `python serve_mcp.py` as a stdio server.
The three tools are quality assessment, bounded candidate generation and object
detection. MCP has its own image registry; it does not restore web sessions.
Input images must be inside the configured data root. Missing detector weights
remain an explicit error. See `docs/14_MCP_SERVER.md` for the protocol demo.

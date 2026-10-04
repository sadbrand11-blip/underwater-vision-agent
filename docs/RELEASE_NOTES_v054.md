# v0.5.4-public.1

Simpler daily use with the existing visual, agent, RAG, memory, LangGraph and MCP capabilities retained.

- A 409-word homepage and an 18-second recording of real local tools; the scheduler is explicitly offline and the brightness change is simulated.
- A compact three-step workbench with collapsed advanced settings and evidence. Failures and unavailable resources stay visible.
- 32 experimental commands moved to `python -m optical_agent.experiments <command>`; native and LangGraph share turn mechanics while retaining their execution structures.
- Independent `public-sanitized-v2` knowledge cards; homepage edits no longer change retrieval sources.
- 26/26 pre/post deterministic records match; 292 Python checks pass, 2 skip, and 27 UI checks pass. The clean CPU/no-key/no-weights bootstrap passes 8 checks.

No paid API requests or retraining. Measured CPU startup is approximately unchanged. Offline equivalence does not establish improved real LLM performance or visual accuracy. Historical evaluation denominators and versions remain unchanged.

[Verification and limitations](SIMPLIFICATION.md) · [Command migration](CLI_MIGRATION.md) · [Chinese Quick Start](QUICK_START_ZH.md)

The detector remains the existing SHA256-verified asset in [v0.5.3-public.1](https://github.com/sadbrand11-blip/underwater-vision-agent/releases/tag/v0.5.3-public.1); this release does not duplicate weights. Source is MIT; sample/GIF imagery is attributed SODD, CC BY 4.0.

# How to read the evidence

The homepage deliberately keeps separate experiments separate.

| Record | Denominator | Interpretation |
|---|---|---|
| Formal legacy Agent Eval | 20 predefined held-out cases repeated three times: 60 runs / 78 turns | 59 completed runs, not 59 independent tasks; 0.2.3 behavior |
| Adaptive development visual demo | Three source images, two runs each | 3/6 complete versus fixed workflow 6/6; developmental limitations remain |
| Frozen C1 retrieval | 40 answerable and 10 no-answer held-out questions | Delivered/filter-aware Recall@3 35.0% vs 47.5%; MRR@10 .3625 vs .4875 |
| MCP protocol comparison | Nine directly compared fields/checks on an actual stdio demo | Interface parity, not recognition accuracy |

RAG no-answer wrong-return counts were TF-IDF 1/10 and Hybrid 0/10. The unfiltered
TF-IDF and Hybrid Recall@3 were both 72.5%; the delivered-ranking improvement
includes development-set rejection thresholds. The paired improvement interval
included zero. These are small benchmarks, not proof of general superiority.

Labels were automatically constructed and reviewed by same-model-family agents;
they have not received external human annotation review. Those audits are
provisional. Repetition measures variability but does not increase task diversity.

The default model's older 40-image SODD pilot had 67.8% raw recall and 55.9%
accepted-result recall; see the [underwater pilot](reports/UNDERWATER_RESULTS.md). This is distinct from Agent
scheduling and RAG retrieval. It is not a claim about unseen underwater domains.

Original Chinese reports and aggregate machine-readable summaries remain in this
snapshot. Raw session traces, private memory, datasets and frozen embedding
assets are excluded. Downloading the full source does not recreate historical
cloud experiments automatically, and no new paid requests were made for export.

The public corpus identity is separate from frozen C1. Its sources are stable
knowledge cards and versioned reports; the homepage is excluded. Current default TF-IDF is a lightweight
delivery configuration, not a newly benchmarked RAG winner.

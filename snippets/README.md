# Representative code

Self-contained excerpts pulled from the live system, lightly trimmed so each
reads on its own. They're chosen to show *judgment*, not volume — the design
decision behind each one is the point. Full context for each lives in the
[`../docs/`](../docs) write-ups.

The running system is private (it holds credentials and personal data), so these
are representative slices, not the whole tree.

| Snippet | What it shows | Write-up |
|---|---|---|
| [`rate_limit_governor.py`](rate_limit_governor.py) | An async token-bucket limiter that *halves its own rate on a ban signal* and creeps back on success — adaptive scraping without getting blocked. | [dropship-intel](../docs/dropship-intel.md) |
| [`route_policy.py`](route_policy.py) | The zero-cost, fail-open routing rule that decides local-vs-frontier per request in microseconds, before any model is touched. | [agent-architecture](../docs/agent-architecture.md) |
| [`deterministic_scorer.py`](deterministic_scorer.py) | Ranking owned by explicit math with hard disqualifiers — the LLM only tags, it never decides the number. | [dropship-intel](../docs/dropship-intel.md) |
| [`walk_forward_cv.py`](walk_forward_cv.py) | Expanding-window time-series cross-validation — the discipline that keeps future information out of a trading model's evaluation. | [trading-system](../docs/trading-system.md) |
| [`moe_offload_launch.sh`](moe_offload_launch.sh) | The llama.cpp invocation that fits a 35B model on 12 GB: attention on the GPU, expert FFNs in system RAM. | [llm-infra](../docs/llm-infra.md) |

Every snippet is the real code, with secrets and machine-specific paths removed.

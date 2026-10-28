# Run chains (historical record)

Every bash chain that launched an experiment from 2026-10-09 to 2026-10-27, copied from the original agent session's
scratchpad (paths made repo-relative). They are the exact recipes behind the results in docs/FINDINGS.md: environment
variables, episode ranges, tags, shard counts, audits, analysis calls.

- Run them from the repo root. `kill -0 <PID>` wait-loops reference processes of the original session and return at once now.
- `irrm_lib.sh` holds `cl()`, the generic closed-loop dev runner (tag, exec noise, imagined noise, value, barrier gamma).
- Newest first in importance: `irrm_lib.sh`, `irrm_a3_chain.sh`, `irrm_a4_chain.sh`, `irrm_cl_chain.sh`, `irrmc_chain.sh`,
  `lsfpub_chain.sh`, `lsfpubc_chain.sh`, `n1f_chain.sh`, `n1m_chain.sh`, `to2x_chain.sh`, `lat_chain.sh`, `h2h4_chain.sh`.
- The next experiment (decisive test H2H5) has its own runner: `scripts/run_h2h5.sh`.

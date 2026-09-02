# Cutegen ChatGPT export bundle

Upload these files to ChatGPT. Start with `cutegen_canvas_payload.json` for full data.

## Files

- `cutegen_canvas_payload.json` — full structured data (speedups, tokens, metadata)
- `cutegen_error_payload.json` — full error/debug data per experiment
- `speedup_curves.csv` — flattened speedups (743 rows: experiment × kernel)
- `token_usage_by_experiment.csv` — LLM tokens per experiment (27 rows)
- `error_summary_by_experiment.csv` — error counts per experiment (28 rows)
- `precision_io_cheats.json` — 2 known I/O precision cheats to exclude
- `README.md` — this guide

## Schema (canvas JSON)

- `exps[]` — experiment configs (backend, profiling method, model, save path)
- `kernels[]` — kernel id, name, type, cohort (sample19 / new8)
- `curves[col][kernel_id]` — speedup vs PyTorch ref at depths 0–10 (`null` = no pass)
- `tokens[col]` — aggregated LLM token usage per experiment
- `precisionCheats[]` — known I/O precision shortcuts; exclude for fair comparisons

## Suggested prompt

> Attached: cutegen export bundle. `speedup_curves.csv` has one row per experiment×kernel.
> `precision_io_cheat=true` rows should be excluded from fair speedup rankings.
> [Your question here]

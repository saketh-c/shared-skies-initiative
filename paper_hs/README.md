# NeurIPS High School Projects 2024-style paper — Shared Skies Initiative

Rebuilt from real, computed results (no fabrication, no placeholders).

- `main.tex` — paper source (4 content pages + references).
- `main.pdf` — compiled output (built with tectonic).
- `neurips_2024.sty` — official NeurIPS High School Projects 2024 style.
- `references.bib` — bibliography.
- `figures/fig_ej.pdf`, `figures/fig_fairness.pdf` — generated from real
  per-sensor results (paper/figures/results.json), not synthetic.

Every number traces to a real artifact: the 240-sensor benchmark from
`paper/figures/results.json`; the deployed v6 model from `models/metrics.json`
and `models/loso_oof.npz`. Compile with `tectonic main.tex` or on Overleaf
(upload the whole folder).

NOTE: NeurIPS has not offered a High School track for 2025 or 2026 (it ran only
in 2024). This uses the 2024 HS format so it is ready for a returning HS track,
an ML workshop, or a science-fair / ISEF submission.

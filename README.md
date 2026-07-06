# MMAS — honest evaluation package

Reframes and re-tests the "beats binary search" claim with a compiled benchmark,
mixed hit/miss queries, and REAL data. Also adds the piecewise-linear extension.

## Files
- **mmas_paper_honest.docx** — the reframed paper (numbers auto-filled from results.csv).
- **bench.cpp** — the honest benchmark (MSVC /O2). Implements stl_lb, binary, interpolation,
  mmas_closed (5-sample linear/power/exp detect + inversion + capped residual), and
  pgm_pla (GreedyPLR piecewise-linear index). Counts probes AND times ns/query; verifies
  every query against std::binary_search (0 errors).
- **piecewise.py** — standalone piecewise-linear model (build_pla + pla_search).
- **mmas_with_piecewise.py** — your mmas.py with the piecewise model wired in as a new
  `'piecewise'` dispatch branch. detect_shape() now falls back to a bounded-error PLA
  (instead of pure binary) when no closed-form model fits and n >= 4096. Validated: 0
  errors on mixed hit/miss queries; on irregular/adversarial data it picks 'piecewise'
  and beats binary on probes (e.g. 12.1 vs 16.9; 8.7 vs 14.6), while analytic curves
  still take the closed-form 1-probe path.
- **gen_data.py / gen_real.py** — dataset generators. Real data = file sizes + mtimes
  harvested from the local filesystem.
- **results.csv** — raw results. **results_readable.txt** — formatted tables + the
  expensive-probe (RAM/SSD/network) extrapolation.
- **make_docx.py / analyze.py** — reporting scripts.

## Headline findings (all queries 50% absent; 0 correctness errors)
1. **Plain interpolation search is dangerous on real data** — up to ~4,000x slower than
   binary on heavy-tailed file sizes, ~9,000x on exponential. Never use it uncapped.
2. **mmas_closed's "1 probe" only happens on synthetic analytic curves.** On every real
   dataset its detector returns "binary" and it matches binary exactly — safe, no gain.
3. **pgm_pla (piecewise) is the real win.** ~11 probes vs binary's ~16; it beats binary on
   in-RAM wall-clock on the larger real dataset and on adversarial data, and dominates once
   probes are expensive (disk/network).
4. **Binary search is the robust default** — never catastrophic, fastest on small arrays.

## Reproduce
```
py gen_data.py           # synthetic + real datasets -> data/
py gen_real.py           # (optional) re-harvest real data only
build.bat bench.cpp bench.exe
bench.exe data 3 32      # dir, repeats, PLA epsilon
py analyze.py results.csv
py make_docx.py results.csv mmas_paper_honest.docx
```

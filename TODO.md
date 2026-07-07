# Roadmap / To-Do

Next steps for this project, from "solid portfolio piece" toward a real publication.
Ordered so each step builds the foundation for the next.

## Done
- [x] Honest compiled benchmark (wall-clock ns/query + probe counts, mixed hit/miss queries)
- [x] Test on real data (filesystem file sizes + mtimes)
- [x] Piecewise-linear (PGM-style) model wired into `mmas.py` as a `'piecewise'` branch
- [x] Assertion tests for the piecewise branch (`test_piecewise.py`, 14/14 passing)
- [x] Reframed paper (`mmas_paper_honest.docx`) + blog writeup (`writeup.md`)
- [x] Public repo with MIT LICENSE

## Foundation (do this first — enables everything below)
- [ ] Integrate with the **SOSD** benchmark (Marcus et al., VLDB 2021) — standard datasets + harness
- [ ] Add **real learned-index baselines** to compare against, not just my own binary/interp:
      **PGM-index** (C++ lib), **RMI**, **RadixSpline**, optionally **ALEX**
- [ ] Re-run at **million-scale on real data with duplicates** (the earlier run got deduped small)
- [ ] Report build time + space + query latency, not just probes

## Direction B — health/bio application (recommended paper path)
- [ ] Find a real biomedical indexing problem with a domain mentor (genomic positions,
      k-mer / interval queries, large record-store lookup)
- [ ] Apply the piecewise/learned index to real data; measure real speedups
- [ ] Draft as an **application paper** (novelty = domain + real data + real speedup)
- [ ] Target venues: BIBM, ISMB workshops, BMC Bioinformatics, or preprint + artifact

## Direction A — learned indexes under a real I/O cost model (workshop path)
- [ ] Benchmark on **real SSD/NVMe / remote I/O**, not just in-memory
- [ ] Co-design the index for I/O: align segments to page/block boundaries;
      choose epsilon to minimize expected I/O, not probe count
- [ ] Include the closed-form-vs-piecewise **space-saving dispatcher** as a section
- [ ] Target venues: DaMoN or aiDM (SIGMOD workshops)

## Meta (the biggest unlock)
- [ ] **Recruit a faculty advisor / co-author** (AUC CS systems-or-algorithms prof, or a
      Harvard/Orange mentor) — turns "student project" into a credible publication
- [ ] Write a one-page research proposal for Direction B to pitch to that advisor

## Housekeeping
- [ ] Add a `Makefile` / build script so `bench.cpp` builds on non-MSVC toolchains (g++/clang)
- [ ] Add a short "Reproduce" GitHub Action (build + run tests on push)
- [ ] Post `writeup.md` externally (Medium / dev.to / personal site) and link it from the CV

---
*Reality check: the goal is a real, peer-reviewed workshop or bioinformatics paper with a
faculty co-author and a rigorous open-source artifact — not "groundbreaking." That is
achievable and outstanding for an undergrad CV.*

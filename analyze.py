"""Format bench.exe CSV into readable tables + expensive-probe latency extrapolation."""
import csv, sys, collections

rows = list(csv.DictReader(open(sys.argv[1])))
by_ds = collections.OrderedDict()
for r in rows:
    by_ds.setdefault(r["dataset"], []).append(r)

def f(x):
    return float(x) if x not in ("", None) else 0.0

def _fmt(ns):
    if ns >= 1e6: return f"{ns/1e6:.2f}ms"
    if ns >= 1e3: return f"{ns/1e3:.1f}us"
    return f"{ns:.0f}ns"

# per-probe cost models (ns) for the three regimes
REGIMES = [("RAM-miss", 100.0), ("SSD", 50_000.0), ("Network", 1_000_000.0)]

print("="*118)
print("IN-RAM WALL-CLOCK (measured) + PROBE COUNTS")
print("="*118)
hdr = f"{'dataset':<18}{'algo':<13}{'ns/query':>10}{'vs stl_lb':>10}{'data_prb':>10}{'max_prb':>9}{'model_prb':>10}{'err':>5}{'detect':>14}"
for ds, rs in by_ds.items():
    lb = next((f(r["ns_per_query"]) for r in rs if r["algo"]=="stl_lb"), None)
    print("-"*118)
    print(f"{ds}  (n={int(rs[0]['n']):,})")
    print(hdr)
    for r in rs:
        ns = f(r["ns_per_query"])
        ratio = f"{ns/lb:.2f}x" if lb else ""
        det = r["detected"]
        print(f"{'':<18}{r['algo']:<13}{ns:>10.1f}{ratio:>10}"
              f"{f(r['mean_data_probes']):>10.2f}{f(r['max_data_probes']):>9.0f}"
              f"{f(r['mean_model_probes']):>10.2f}{r['errors'] or '0':>5}{det:>14}")

print()
print("="*118)
print("EXTRAPOLATED LATENCY WHEN A PROBE IS EXPENSIVE (total probes x per-probe cost)")
print("total probes = data_probes + model_probes ; lower is better; winner in [brackets]")
print("="*118)
for ds, rs in by_ds.items():
    print("-"*118)
    print(f"{ds}")
    cand = [r for r in rs if r["algo"] in ("binary","interp","mmas_closed","pgm_pla")]
    print(f"{'algo':<14}{'tot_probes':>11}" + "".join(f"{name:>14}" for name,_ in REGIMES))
    best = {name: min(f(r['mean_data_probes'])+f(r['mean_model_probes']) for r in cand) for name,_ in REGIMES}
    for r in cand:
        tot = f(r['mean_data_probes'])+f(r['mean_model_probes'])
        cells = []
        for name, cost in REGIMES:
            lat = tot*cost
            mark = "[%s]"%_fmt(lat) if abs(tot-best[name])<1e-9 else _fmt(lat)
            cells.append(f"{mark:>14}")
        print(f"{r['algo']:<14}{tot:>11.2f}" + "".join(cells))

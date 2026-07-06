"""Harvest REAL sorted data from the filesystem; reuse existing synthetic .bin files."""
import os, struct, random, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = random.Random(20260706)
N_QUERIES = 200_000

def write_i64(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr)))
        f.write(struct.pack("<%dq" % len(arr), *arr))

def make_queries(arr):
    n = len(arr); present = set(arr); q = []
    for _ in range(N_QUERIES // 2): q.append(arr[rng.randrange(n)])
    tries = 0
    while len(q) < N_QUERIES and tries < N_QUERIES * 20:
        tries += 1; i = rng.randrange(n - 1); lo, hi = arr[i], arr[i + 1]
        if hi - lo >= 2:
            c = rng.randrange(lo + 1, hi)
            if c not in present: q.append(c)
    while len(q) < N_QUERIES: q.append(arr[rng.randrange(n)])
    rng.shuffle(q); return q

def emit(name, arr):
    # REAL arrays: keep duplicates (real sorted data repeats); just sort.
    arr = sorted(int(v) for v in arr)
    if len(arr) < 100: print("  SKIP", name, len(arr)); return None
    write_i64(os.path.join(OUT, name + ".bin"), arr)
    write_i64(os.path.join(OUT, name + ".qry"), make_queries(arr))
    print(f"  {name:16s} n={len(arr):>9,} min={arr[0]:>14,} max={arr[-1]:>16,}")
    return name

roots = [r"C:\Windows\System32", r"C:\Python314", r"C:\Users\Omar\AppData\Local"]
sizes, mtimes = [], []
CAP = 700_000
t0 = time.time()
for root in roots:
    if len(sizes) >= CAP: break
    for dp, dn, fns in os.walk(root, onerror=lambda e: None):
        for fn in fns:
            try:
                st = os.stat(os.path.join(dp, fn)); sizes.append(st.st_size); mtimes.append(int(st.st_mtime))
            except OSError: pass
        if len(sizes) >= CAP: break
print(f"  harvested {len(sizes):,} files in {time.time()-t0:.0f}s")
emit("real_filesizes", sizes)
emit("real_mtimes", mtimes)

# rebuild full manifest (synthetic already on disk)
syn = ["syn_linear","syn_quadratic","syn_exponential","syn_uniform","syn_adversarial"]
real = ["real_filesizes","real_mtimes"]
present = [n for n in syn+real if os.path.exists(os.path.join(OUT, n+".bin"))]
open(os.path.join(OUT,"manifest.txt"),"w").write("\n".join(present)+"\n")
print("manifest:", present)

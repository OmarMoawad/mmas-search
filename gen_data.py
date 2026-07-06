"""
Generate benchmark datasets as sorted, unique int64 binary files (.bin, little-endian).
Includes SYNTHETIC shapes and REAL data harvested from the local filesystem.
Also writes a query file per dataset: 50% hits (present) + 50% misses (absent, in-range).
"""
import os, struct, random, math, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(20260706)

N_SYNTH = 1_000_000
N_QUERIES = 200_000


def write_i64(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(arr)))
        f.write(struct.pack("<%dq" % len(arr), *arr))


def dedup_sort(vals):
    return sorted(set(int(v) for v in vals))


def make_queries(arr):
    """50% hits drawn from arr, 50% misses (values strictly between neighbors, absent)."""
    n = len(arr)
    present = set(arr)
    q = []
    half = N_QUERIES // 2
    for _ in range(half):
        q.append(arr[rng.randrange(n)])
    tries = 0
    while len(q) < N_QUERIES and tries < N_QUERIES * 20:
        tries += 1
        i = rng.randrange(n - 1)
        lo, hi = arr[i], arr[i + 1]
        if hi - lo >= 2:
            cand = rng.randrange(lo + 1, hi)
            if cand not in present:
                q.append(cand)
    # pad with hits if not enough gaps
    while len(q) < N_QUERIES:
        q.append(arr[rng.randrange(n)])
    rng.shuffle(q)
    return q


def emit(name, arr):
    arr = dedup_sort(arr)
    if len(arr) < 100:
        print(f"  SKIP {name}: only {len(arr)} unique values")
        return
    write_i64(os.path.join(OUT, name + ".bin"), arr)
    write_i64(os.path.join(OUT, name + ".qry"), make_queries(arr))
    print(f"  {name:22s} n={len(arr):>10,}  min={arr[0]:>16,}  max={arr[-1]:>18,}")
    return name


manifest = []

print("SYNTHETIC:")
manifest.append(emit("syn_linear",      [i for i in range(N_SYNTH)]))
manifest.append(emit("syn_quadratic",   [i * i for i in range(N_SYNTH)]))
manifest.append(emit("syn_exponential", [int(round(2 ** (50 * i / N_SYNTH))) for i in range(N_SYNTH)]))
manifest.append(emit("syn_uniform",     rng.sample(range(20 * N_SYNTH), N_SYNTH)))
# adversarial: two dense clusters far apart (interpolation's nemesis)
adv = [rng.randint(0, N_SYNTH // 20) for _ in range(N_SYNTH // 2)] + \
      [rng.randint(400 * N_SYNTH, 400 * N_SYNTH + N_SYNTH // 20) for _ in range(N_SYNTH // 2)]
manifest.append(emit("syn_adversarial", adv))

print("REAL (harvesting filesystem, this may take a moment)...")
roots = [r"C:\Windows\System32", r"C:\Python314", r"C:\Users\Omar\AppData\Local"]
sizes, mtimes = [], []
CAP = 700_000
for root in roots:
    if len(sizes) >= CAP:
        break
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        for fn in filenames:
            try:
                st = os.stat(os.path.join(dirpath, fn))
                sizes.append(st.st_size)
                mtimes.append(int(st.st_mtime))
            except OSError:
                pass
        if len(sizes) >= CAP:
            break
print(f"  harvested {len(sizes):,} files")
manifest.append(emit("real_filesizes", sizes))   # heavy-tailed (log-normal-ish)
manifest.append(emit("real_mtimes",    mtimes))  # unix timestamps, clustered

manifest = [m for m in manifest if m]
with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write("\n".join(manifest) + "\n")
print("manifest:", manifest)

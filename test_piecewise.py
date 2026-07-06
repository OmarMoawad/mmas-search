"""Assertion tests for the piecewise-linear branch of MMAS.
Run: py test_piecewise.py   (exits non-zero on any failure)"""
import math
import random
import statistics
import bisect
from mmas_with_piecewise import (binary_search, mmas_search, detect_shape,
                                  build_pla, pla_search)


def make_dataset(kind, n, seed=0):
    rng = random.Random(seed)
    if kind == "lognormal":       # heavy-tailed, real-ish (file sizes / durations)
        return sorted(int(rng.lognormvariate(11, 2.2)) for _ in range(n))
    if kind == "adversarial":     # two far-apart dense clusters
        return sorted([rng.randint(0, 5000) for _ in range(n // 2)] +
                      [rng.randint(10**7, 10**7 + 5000) for _ in range(n - n // 2)])
    if kind == "piecewise_slopes":  # three linear regimes with different slopes
        a = [3 * i for i in range(n // 3)]
        b = [a[-1] + 50 * (i + 1) for i in range(n // 3)]
        c = [b[-1] + 2 * (i + 1) for i in range(n - 2 * (n // 3))]
        return a + b + c
    raise ValueError(kind)


def mixed_queries(A, n_queries, seed=99):
    """50% present, 50% absent (strictly between neighbors)."""
    rng = random.Random(seed)
    present = [A[rng.randrange(len(A))] for _ in range(n_queries // 2)]
    absent, tries = [], 0
    while len(absent) < n_queries - len(present) and tries < n_queries * 40:
        tries += 1
        i = rng.randrange(len(A) - 1)
        if A[i + 1] - A[i] >= 2:
            absent.append(A[i] + 1)
    q = present + absent
    rng.shuffle(q)
    return q


def truth(A, v):
    j = bisect.bisect_left(A, v)
    return j if (j < len(A) and A[j] == v) else -1


PASS = 0
def check(cond, msg):
    global PASS
    assert cond, "FAIL: " + msg
    PASS += 1
    print("  ok:", msg)


EPS = 32
print("=== 1. build_pla honours its error bound (|pred - actual| <= eps) ===")
for kind in ("lognormal", "adversarial", "piecewise_slopes"):
    A = make_dataset(kind, 60_000, seed=1)
    keys, slopes, y0s = build_pla(A, eps=EPS)
    worst = 0
    for j, x in enumerate(A):
        s = bisect.bisect_right(keys, x) - 1
        if s < 0:
            s = 0
        pred = y0s[s] + round(slopes[s] * (x - keys[s]))
        worst = max(worst, abs(pred - j))
    check(worst <= EPS, f"{kind}: max prediction error {worst} <= eps={EPS} "
                        f"({len(keys)} segments)")

print("\n=== 2. detect_shape picks 'piecewise' on irregular data (n >= 4096) ===")
for kind in ("lognormal", "adversarial", "piecewise_slopes"):
    A = make_dataset(kind, 60_000, seed=2)
    (model, params), probes = detect_shape(A)
    check(model == "piecewise", f"{kind}: detected '{model}' (expected piecewise)")

print("\n=== 3. correctness: every hit found, every miss reported, 0 errors ===")
for kind in ("lognormal", "adversarial", "piecewise_slopes"):
    A = make_dataset(kind, 60_000, seed=3)
    shape_info, _ = detect_shape(A)
    q = mixed_queries(A, 4000)
    errs = 0
    for v in q:
        idx, _ = mmas_search(A, v, shape_info=shape_info, fast_path=False)
        t = truth(A, v)
        got_ok = (idx >= 0 and A[idx] == v) if idx >= 0 else True
        if (idx >= 0) != (t >= 0) or not got_ok:
            errs += 1
    check(errs == 0, f"{kind}: {errs} errors over {len(q)} mixed queries "
                     f"(model={shape_info[0]})")

print("\n=== 4. piecewise beats binary on probes for irregular data ===")
for kind in ("lognormal", "adversarial"):
    A = make_dataset(kind, 60_000, seed=4)
    shape_info, _ = detect_shape(A)
    q = mixed_queries(A, 3000)
    mm = [mmas_search(A, v, shape_info=shape_info, fast_path=False)[1] for v in q]
    bb = [binary_search(A, v)[1] for v in q]
    check(statistics.mean(mm) < statistics.mean(bb),
          f"{kind}: mmas {statistics.mean(mm):.2f} < binary {statistics.mean(bb):.2f} probes")

print("\n=== 5. worst-case bound: piecewise probes <= log2(#segs)+2*log2(2*eps+3)+5 ===")
A = make_dataset("lognormal", 60_000, seed=5)
(model, params), _ = detect_shape(A)
pla, eps = params
bound = len(pla[0]).bit_length() + 2 * math.ceil(math.log2(2 * eps + 3)) + 5
q = mixed_queries(A, 4000)
maxp = max(pla_search(A, v, pla, eps=eps)[1] for v in q)
check(maxp <= bound, f"max probes {maxp} <= bound {bound}")

print("\n=== 6. graceful fallback: tiny array does NOT use piecewise ===")
small = sorted(random.Random(6).sample(range(10**6), 1000))  # n < 4096
(model, _), _ = detect_shape(small)
check(model in ("binary", "linear", "power", "exponential") and model != "piecewise",
      f"n=1000 -> model '{model}' (not piecewise)")

print("\n=== 7. closed-form still preferred where an exact law fits ===")
cube = [i ** 3 for i in range(60_000)]
(model, _), _ = detect_shape(cube)
check(model == "power", f"cubic -> '{model}' (closed-form, not piecewise)")

print(f"\nALL {PASS} ASSERTIONS PASSED")

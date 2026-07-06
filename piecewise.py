"""
Piecewise-linear model for MMAS — the extension that makes it work on REAL data.

Instead of fitting ONE global closed-form curve (linear/power/exp), we approximate
the key->position function by a chain of straight segments, each guaranteed to
predict any contained key's position within +/- EPS. This is the PGM-index idea
(Ferragina & Vinciguerra, 2020) in ~40 lines, no ML.

Drop-in for mmas.py:
  - build_pla(A, eps) is a ONE-TIME index build (amortized like detect_shape).
  - pla_search(A, v, pla) returns (idx, probes) exactly like the other searches.
  - Fallback safety: the windowed binary search bounds worst case to
    O(log(#segments) + log(eps)); if the model is bad, #segments grows and it
    degrades gracefully toward binary search — never worse asymptotically.

How it slots into the existing dispatcher: add 'piecewise' as a model kind.
detect_shape() first tries the 3 closed-form fits (cheap, 5 probes). If none
fit AND the array is large & monotonic, build a PLA once and dispatch to it
instead of falling straight back to pure binary search.
"""
import bisect


def build_pla(A, eps=32):
    """GreedyPLR bounded-error segmentation.
    Returns (seg_keys, seg_slope, seg_y0): parallel lists, one entry per segment.
    Guarantee: for every key A[j] in a segment, |predict(A[j]) - j| <= eps.
    """
    n = len(A)
    keys, slopes, y0s = [], [], []
    i = 0
    while i < n:
        x0, y0 = A[i], i
        slo, shi = -float('inf'), float('inf')
        j = i + 1
        while j < n:
            dx = A[j] - x0
            if dx == 0:            # duplicate key: keep in segment
                j += 1
                continue
            lo = ((j - y0) - eps) / dx
            hi = ((j - y0) + eps) / dx
            nlo, nhi = max(slo, lo), min(shi, hi)
            if nlo > nhi:
                break
            slo, shi = nlo, nhi
            j += 1
        slope = 0.0 if slo == -float('inf') or shi == float('inf') else 0.5 * (slo + shi)
        keys.append(x0); slopes.append(slope); y0s.append(y0)
        i = j
    return keys, slopes, y0s


def pla_search(A, v, pla, eps=32):
    """Search using a prebuilt PLA. Returns (idx, probes) counting DATA-array probes.
    (Segment lookup touches a small, cache-resident model array; counted too, honestly.)"""
    keys, slopes, y0s = pla
    n = len(A)
    probes = 0

    # locate segment: rightmost seg with key <= v  (binary search over anchors)
    s = bisect.bisect_right(keys, v) - 1
    if s < 0:
        s = 0
    probes += max(1, (len(keys)).bit_length())   # ~log2(#segments) model probes

    pred = y0s[s] + int(round(slopes[s] * (v - keys[s])))
    lo = max(0, pred - eps - 1)
    hi = min(n - 1, pred + eps + 1)

    # windowed binary search in the data
    while lo <= hi:
        mid = (lo + hi) // 2
        probes += 1
        if A[mid] == v:
            return mid, probes
        if A[mid] < v:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, probes


if __name__ == "__main__":
    # smoke test: real-ish irregular data that NO closed-form model fits
    import random, math
    rng = random.Random(1)
    A = sorted(set(int(rng.lognormvariate(10, 2)) for _ in range(200_000)))
    pla = build_pla(A, eps=32)
    ok = 0
    probes_total = 0
    for _ in range(5000):
        v = A[rng.randrange(len(A))]
        idx, p = pla_search(A, v, pla)
        probes_total += p
        ok += (idx >= 0 and A[idx] == v)
    print(f"n={len(A):,} segments={len(pla[0]):,} "
          f"correct={ok}/5000 mean_probes={probes_total/5000:.2f} "
          f"vs binary ~{math.log2(len(A)):.1f}")

"""Phase 2: strong vs weak scaling with a serial Stage A and a parallel Stage B (persistent Pool)."""
import math
import time
import multiprocessing as mp
from harness import (detect_pcores, run_benchmark, emit_verify_tag, save_section, get_params,
                     seed_value, QUICK, quick_banner)

_B = None


def _init(barrier):
    global _B
    _B = barrier


def stage_a(n):
    """Serial stage: 10 math ops (5 sin + 5 sqrt) per element."""
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(n):
        x = i * 1e-3 + 0.5
        for _ in range(5):
            x = sqrt(1.0 + abs(sin(x)))
        acc += x
    return acc


def stage_b(rng):
    """Parallel stage: 30 FMA/trig ops per element (10 x [FMA, sin, FMA]), no shared writes."""
    _B.wait()                      # makes sure every worker takes exactly one chunk
    lo, hi = rng
    sin = math.sin
    acc = 0.0
    for i in range(lo, hi):
        x = i * 1e-6 + 0.1
        for _ in range(10):
            x = x * 0.999999 + 0.5
            s = sin(x)
            x = s * s + x * 0.5
        acc += x
    return acc


def measure(p, N, warm, runs):
    nA = int(0.05 * N)
    ranges = [(N * k // p, N * (k + 1) // p) for k in range(p)]
    barrier = mp.Barrier(p)
    tA = []
    with mp.Pool(processes=p, initializer=_init, initargs=(barrier,)) as pool:   # created OUTSIDE timed loop
        pool.map(stage_b, [(0, 10)] * p, chunksize=1)                              # warm the pool
        def fn():
            t0 = time.perf_counter()
            stage_a(nA)                                                            # Stage A: serial, parent
            tA.append((time.perf_counter() - t0) * 1e3)
            pool.map(stage_b, ranges, chunksize=1)                                 # Stage B: parallel
        st = run_benchmark(fn, warm, runs)
    a_med = sorted(tA[-runs:])[runs // 2]
    return st, a_med


def main():
    quick_banner()
    pphys, plog = get_params()
    N = 120_000 if QUICK else 1_200_000
    N0 = 20_000 if QUICK else 200_000
    warm, runs = (1, 3) if QUICK else (2, 7)
    print(f"P_phys={pphys}  P_log={plog}  N={N}  N0={N0}")

    # ---------------- strong scaling ----------------
    strong, t1, a1 = [], None, None
    print("\n== Strong scaling (N fixed) ==")
    print(f"{'p':>3} {'T_p ms':>10} {'IQR':>8} {'S_p':>6} {'E_p':>6} {'T_A ms':>8}")
    for p in range(1, plog + 1):
        st, a = measure(p, N, warm, runs)
        if p == 1:
            t1, a1 = st.median_ms, a
        S = t1 / st.median_ms
        strong.append({"p": p, "T": st.median_ms, "iqr": st.iqr_ms, "S": S, "E": S / p, "tA": a})
        print(f"{p:>3} {st.median_ms:>10.1f} {st.iqr_ms:>8.1f} {S:>6.2f} {S / p:>6.2f} {a:>8.1f}")
    s_frac = a1 / t1
    s_inf = 1.0 / s_frac
    s_plog = strong[-1]["S"]
    print(f"\nserial fraction s = T_A(1)/T_1 = {s_frac:.4f}  -> Amdahl cap S_inf = {s_inf:.1f}")
    print(f"observed S({plog}) = {s_plog:.2f}  ({100 * s_plog / s_inf:.1f}% of the Amdahl cap; "
          f"Amdahl bound at p={plog}: {1 / (s_frac + (1 - s_frac) / plog):.2f})")

    # ---------------- weak scaling ----------------
    weak, tw1 = [], None
    print("\n== Weak scaling (N = p * N0) ==")
    print(f"{'p':>3} {'N':>9} {'T_p ms':>10} {'E_weak':>7} {'alpha':>7} {'S_Gust':>7}")
    for p in range(1, plog + 1):
        st, a = measure(p, p * N0, warm, runs)
        if p == 1:
            tw1 = st.median_ms
        ew = tw1 / st.median_ms
        alpha = a / st.median_ms
        sg = p - alpha * (p - 1)
        weak.append({"p": p, "N": p * N0, "T": st.median_ms, "iqr": st.iqr_ms, "Ew": ew,
                     "alpha": alpha, "SG": sg, "tA": a})
        print(f"{p:>3} {p * N0:>9} {st.median_ms:>10.1f} {ew:>7.2f} {alpha:>7.3f} {sg:>7.2f}")

    # ---------------- architectural inflection ----------------
    infl = None
    S = {r["p"]: r["S"] for r in strong}
    pc = detect_pcores()
    # boundary: SMT (Pphys->Pphys+1) if Plog>Pphys, else P-core -> E-core (hybrid CPU, e.g. Apple M-series)
    if plog > pphys >= 1:
        bnd, kind = pphys, "SMT"
    elif pc and 1 <= pc < plog:
        bnd, kind = pc, "P->E cores"
    else:
        bnd, kind = None, None
    if bnd:
        before = (S[bnd] - S[bnd - 1]) if bnd >= 2 else S[bnd]
        after = S[bnd + 1] - S[bnd]
        avg_before = (S[bnd] - S[1]) / (bnd - 1) if bnd > 1 else S[1]
        drop = 100 * (1 - after / before) if before > 0 else float("nan")
        infl = {"pphys": pphys, "boundary": bnd, "kind": kind, "dS_before": before, "dS_after": after,
                "avg_dS_before": avg_before, "drop_pct": drop}
        print(f"\nBoundary {kind} at p={bnd}. Marginal speedup: p={bnd - 1}->{bnd}: {before:+.2f} | "
              f"p={bnd}->{bnd + 1}: {after:+.2f} (drop {drop:.0f}%), mean gain per core below boundary: {avg_before:.2f}")
    else:
        print("\nP_log == P_phys and no hybrid cores detected: no boundary to analyse (set PCORES=<n> to force).")

    tag = emit_verify_tag("PHASE2", seed_value(), strong[-1]["S"], weak[-1]["SG"])
    print("\n" + tag)
    save_section("phase2", {"N": N, "N0": N0, "strong": strong, "weak": weak,
                            "s": s_frac, "S_inf": s_inf, "inflection": infl,
                            "hash": tag, "quick": QUICK})


if __name__ == "__main__":
    main()

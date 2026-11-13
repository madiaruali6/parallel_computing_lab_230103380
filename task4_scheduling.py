"""Phase 4: skewed (triangular) workload - static block, static cyclic and shared-memory dynamic chunk queue."""
import math
import time
import multiprocessing as mp
from harness import (run_benchmark, emit_verify_tag, save_section, get_params,
                     seed_value, QUICK, quick_banner)


def do_items(lo, hi, step):
    """Item i costs K(i) = 1 + floor((i/40)^2) heavy iterations (sin, cos, sqrt)."""
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    acc = 0.0
    for i in range(lo, hi, step):
        K = 1 + int((i / 40.0) ** 2)
        x = 0.5
        for _ in range(K):
            x = sqrt(1.0 + abs(sin(x) * cos(x)))
        acc += x
    return acc


def worker(strategy, t, p, N, C, counter, tarr, barrier):
    barrier.wait()                       # common start
    t0 = time.perf_counter()
    acc = 0.0
    if strategy == "block":
        acc = do_items(t * N // p, (t + 1) * N // p, 1)
    elif strategy == "cyclic":
        acc = do_items(t, N, p)
    else:                                # dynamic chunk queue
        while True:
            with counter.get_lock():
                start = counter.value
                counter.value = start + C
            if start >= N:
                break
            acc += do_items(start, min(start + C, N), 1)
    tarr[t] = time.perf_counter() - t0   # active runtime (includes queue waits)


def measure(strategy, p, N, C, warm, runs):
    counter = mp.Value("i", 0)
    tarr = mp.Array("d", p)
    barrier = mp.Barrier(p)
    lams = []

    def fn():
        counter.value = 0
        procs = [mp.Process(target=worker, args=(strategy, t, p, N, C, counter, tarr, barrier))
                 for t in range(p)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        ts = list(tarr)
        lams.append(max(ts) / (sum(ts) / p))

    st = run_benchmark(fn, warm, runs)
    lam = sorted(lams[-runs:])[runs // 2]
    return st, lam


def main():
    quick_banner()
    pphys, plog = get_params()
    p = pphys
    N = 1600 if QUICK else 8000
    warm, runs = (1, 3) if QUICK else (1, 5)
    total_iters = sum(1 + int((i / 40.0) ** 2) for i in range(N))
    print(f"P_phys={pphys}  N={N}  total heavy iterations = {total_iters:,}  (warmup={warm}, runs={runs})")

    results = {}
    print(f"\n{'strategy':<22} {'Twall ms':>10} {'IQR':>8} {'lambda':>7}")
    for name, strat, C in (("Block static", "block", 1), ("Cyclic static", "cyclic", 1)):
        st, lam = measure(strat, p, N, C, warm, runs)
        results[strat] = {"T": st.median_ms, "iqr": st.iqr_ms, "lam": lam}
        print(f"{name:<22} {st.median_ms:>10.1f} {st.iqr_ms:>8.1f} {lam:>7.3f}")

    dyn = []
    for C in (1, 8, 64, 512):
        st, lam = measure("dynamic", p, N, C, warm, runs)
        dyn.append({"C": C, "T": st.median_ms, "iqr": st.iqr_ms, "lam": lam})
        print(f"{'Dynamic C=' + str(C):<22} {st.median_ms:>10.1f} {st.iqr_ms:>8.1f} {lam:>7.3f}")
    best = min(dyn, key=lambda d: d["T"])
    print(f"\nC* = {best['C']} (lowest wall time {best['T']:.1f} ms, lambda {best['lam']:.3f})")
    print(f"speedup of Dynamic(C*) over Block static: x{results['block']['T'] / best['T']:.2f}")

    tag = emit_verify_tag("PHASE4", seed_value(), results["block"]["T"], best["T"])
    print("\n" + tag)
    save_section("phase4", {"N": N, "p": p, "total_iters": total_iters, "block": results["block"],
                            "cyclic": results["cyclic"], "dynamic": dyn, "c_star": best["C"],
                            "hash": tag, "quick": QUICK})


if __name__ == "__main__":
    main()

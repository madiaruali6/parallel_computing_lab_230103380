"""Phase 3: empirical roofline (DRAM bandwidth + FMA ceiling) and progressive / cache-blocked GEMM."""
import time
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import (emit_verify_tag, save_section, get_params, get_hw, seed_value,
                     stats_from, QUICK, quick_banner)

_G = {}


def _init(barrier, names, N):
    _G["b"] = barrier
    _G["shm"] = {k: shared_memory.SharedMemory(name=v) for k, v in names.items()}
    _G["bwa"] = _G["shm"]["bwa"].buf
    _G["bwb"] = _G["shm"]["bwb"].buf
    for k in ("A", "B", "C1", "C2", "C3"):
        _G[k] = _G["shm"][k].buf[:N * N * 8].cast("d")
    _G["N"] = N


# ------------------------------ Part 3A tasks ---------------------------------
def bw_task(se):
    s, e = se
    _G["b"].wait()
    t0 = time.perf_counter_ns()
    _G["bwa"][s:e] = _G["bwb"][s:e]
    t1 = time.perf_counter_ns()
    return t0, t1


def fma_kernel(iters):
    """Unrolled 16-variable scalar FMA loop: 16 FMA = 32 FLOP per iteration."""
    a0, a1, a2, a3, a4, a5, a6, a7 = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
    a8, a9, a10, a11, a12, a13, a14, a15 = 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
    c = 0.9999999
    d = 1e-9
    for _ in range(iters):
        a0 = a0 * c + d; a1 = a1 * c + d; a2 = a2 * c + d; a3 = a3 * c + d
        a4 = a4 * c + d; a5 = a5 * c + d; a6 = a6 * c + d; a7 = a7 * c + d
        a8 = a8 * c + d; a9 = a9 * c + d; a10 = a10 * c + d; a11 = a11 * c + d
        a12 = a12 * c + d; a13 = a13 * c + d; a14 = a14 * c + d; a15 = a15 * c + d
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15


def fma_task(iters):
    _G["b"].wait()
    t0 = time.perf_counter_ns()
    fma_kernel(iters)
    t1 = time.perf_counter_ns()
    return t0, t1


# ------------------------------ Part 3B kernels --------------------------------
def k_naive(A, B, C, N, r0, r1):                    # Variant 1: i-j-k, strided B[k*N + j]
    for i in range(r0, r1):
        ib = i * N
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[ib + k] * B[k * N + j]
            C[ib + j] = s


def k_reorder(A, B, C, N, r0, r1):                  # Variant 2: i-k-j, a_ik hoisted, unit-stride j
    for i in range(r0, r1):
        ib = i * N
        for j in range(N):
            C[ib + j] = 0.0
        for k in range(N):
            a = A[ib + k]
            kb = k * N
            for j in range(N):
                C[ib + j] += a * B[kb + j]


def k_tiled(A, B, C, N, r0, r1, BS):                # Variant 3: 6 loops (ii,jj,kk,i,k,j)
    for i in range(r0, r1):
        ib = i * N
        for j in range(N):
            C[ib + j] = 0.0
    for ii in range(r0, r1, BS):
        imax = min(ii + BS, r1)
        for jj in range(0, N, BS):
            jmax = min(jj + BS, N)
            for kk in range(0, N, BS):
                kmax = min(kk + BS, N)
                for i in range(ii, imax):
                    ib = i * N
                    for k in range(kk, kmax):
                        a = A[ib + k]
                        kb = k * N
                        for j in range(jj, jmax):
                            C[ib + j] += a * B[kb + j]


def gemm_task(args):
    variant, r0, r1, BS = args
    N, A, B = _G["N"], _G["A"], _G["B"]
    _G["b"].wait()
    t0 = time.perf_counter_ns()
    if variant == 1:
        k_naive(A, B, _G["C1"], N, r0, r1)
    elif variant == 2:
        k_reorder(A, B, _G["C2"], N, r0, r1)
    else:
        k_tiled(A, B, _G["C3"], N, r0, r1, BS)
    t1 = time.perf_counter_ns()
    return t0, t1


# ------------------------------ driver ------------------------------------------
def bench_span(pool, task, args_list, warm, runs):
    """Wall-span of the parallel region = max(end) - min(start) measured inside the workers."""
    samples = []
    for i in range(warm + runs):
        res = pool.map(task, args_list, chunksize=1)
        span_ms = (max(r[1] for r in res) - min(r[0] for r in res)) / 1e6
        if i >= warm:
            samples.append(span_ms)
    return stats_from(samples)


def main():
    quick_banner()
    pphys, plog = get_params()
    hw = get_hw()
    p = pphys
    warm, runs = (1, 3) if QUICK else (2, 7)
    BUF = 8 * 10 ** 6 if QUICK else 64 * 10 ** 6
    N = 64 if QUICK else 256
    print(f"P_phys={pphys}  N={N}  buffer={BUF / 1e6:.0f} MB x2")

    sizes = {"bwa": BUF, "bwb": BUF}
    for k in ("A", "B", "C1", "C2", "C3"):
        sizes[k] = N * N * 8
    shms = {k: shared_memory.SharedMemory(create=True, size=v) for k, v in sizes.items()}
    names = {k: v.name for k, v in shms.items()}
    views = {}
    try:
        for k in ("A", "B", "C1", "C2", "C3"):
            views[k] = shms[k].buf[:N * N * 8].cast("d")
        for i in range(N * N):
            views["A"][i] = ((i * 7 + 3) % 101) / 101.0
            views["B"][i] = ((i * 13 + 5) % 97) / 97.0 - 0.3
        barrier = mp.Barrier(p)
        with mp.Pool(p, initializer=_init, initargs=(barrier, names, N)) as pool:
            # -------- Part 3A: roofline ceilings --------
            bw_args = [(BUF * t // p, BUF * (t + 1) // p) for t in range(p)]
            st = bench_span(pool, bw_task, bw_args, warm, runs)
            bw = 2 * BUF / ((st.median_ms / 1e3) * 1e9)
            print(f"\n[3A] copy: median {st.median_ms:.2f} ms (IQR {st.iqr_ms:.2f}) -> BW_sustained = {bw:.2f} GB/s")

            iters = 60_000_000 // 32            # 6.0e7 FLOP per worker
            flops_total = iters * 32 * p
            st = bench_span(pool, fma_task, [iters] * p, warm, runs)
            pcomp = flops_total / ((st.median_ms / 1e3) * 1e9)
            ridge = pcomp / bw
            print(f"[3A] FMA : median {st.median_ms:.1f} ms (IQR {st.iqr_ms:.1f}) -> P_sustained = {pcomp:.3f} GFLOP/s "
                  f"({p} procs), ridge I* = {ridge:.4f} FLOP/B")

            # -------- Part 3B: GEMM --------
            rows = [(N * t // p, N * (t + 1) // p) for t in range(p)]
            flops = 2 * N ** 3
            gf = lambda ms: flops / ((ms / 1e3) * 1e9)

            st1 = bench_span(pool, gemm_task, [(1, a, b, 0) for a, b in rows], warm, runs)
            st2 = bench_span(pool, gemm_task, [(2, a, b, 0) for a, b in rows], warm, runs)
            g1, g2 = gf(st1.median_ms), gf(st2.median_ms)
            print(f"\n[3B] V1 naive i-j-k    : {st1.median_ms:9.1f} ms  {g1:.4f} GFLOP/s")
            print(f"[3B] V2 reordered i-k-j: {st2.median_ms:9.1f} ms  {g2:.4f} GFLOP/s  (x{st1.median_ms / st2.median_ms:.2f} vs V1)")

            tiles = []
            max_err = 0.0
            print(f"\n{'BS':>4} {'ms':>9} {'IQR':>7} {'GFLOP/s':>9} {'W_tile KB':>10} {'I=BS/12':>8} {'fits':>6}")
            for BS in (8, 16, 32, 64, 96, 128):
                st3 = bench_span(pool, gemm_task, [(3, a, b, BS) for a, b in rows], warm, runs)
                err = max(abs(views["C3"][i] - views["C2"][i]) for i in range(N * N))
                max_err = max(max_err, err)
                w = 3 * BS * BS * 8
                fits = ("L1d" if hw["l1d"] and w <= hw["l1d"] else
                        "L2" if hw["l2"] and w <= hw["l2"] else
                        "L3" if hw["l3"] and w <= hw["l3"] else "?")
                g3 = gf(st3.median_ms)
                tiles.append({"bs": BS, "ms": st3.median_ms, "iqr": st3.iqr_ms, "gflops": g3,
                              "wtile": w, "I": BS / 12.0, "fits": fits, "err": err})
                print(f"{BS:>4} {st3.median_ms:>9.1f} {st3.iqr_ms:>7.1f} {g3:>9.4f} {w / 1024:>10.1f} {BS / 12:>8.2f} {fits:>6}")
            err12 = max(abs(views["C1"][i] - views["C2"][i]) for i in range(N * N))
            ok = max_err < 1e-9 and err12 < 1e-9
            print(f"\nverification: max|C3-C2| = {max_err:.2e}, max|C1-C2| = {err12:.2e} -> {'OK' if ok else 'FAILED'}")
            best = max(tiles, key=lambda t: t["gflops"])
            print(f"BS* = {best['bs']} ({best['gflops']:.4f} GFLOP/s), W_tile = {best['wtile'] / 1024:.1f} KB, "
                  f"I_tiled = {best['I']:.2f} FLOP/B")
            print(f"L1d = {hw['l1d']} B, L2 = {hw['l2']} B")

            att = lambda I: min(pcomp, bw * I)
            print("\nRoofline placement (attainable = min(P_sustained, BW*I)):")
            print(f"  V1 measured {g1:.4f} | V2 measured {g2:.4f} | V3(BS*) measured {best['gflops']:.4f} "
                  f"| attainable at I={best['I']:.2f}: {att(best['I']):.3f} GFLOP/s | compute roof {pcomp:.3f}")

            tag = emit_verify_tag("PHASE3", seed_value(), g1, best["gflops"])
            print("\n" + tag)
            save_section("phase3", {
                "N": N, "buf_bytes": BUF, "bw_gbs": bw, "p_sustained": pcomp, "ridge": ridge,
                "v1": {"ms": st1.median_ms, "iqr": st1.iqr_ms, "gflops": g1},
                "v2": {"ms": st2.median_ms, "iqr": st2.iqr_ms, "gflops": g2},
                "tiles": tiles, "bs_star": best["bs"], "bs_star_gflops": best["gflops"],
                "max_err": max_err, "max_err_v1v2": err12, "verified": ok,
                "hash": tag, "quick": QUICK})
    finally:
        for v in views.values():
            try:
                v.release()
            except Exception:
                pass
        for s in shms.values():
            try:
                s.close()
                s.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()

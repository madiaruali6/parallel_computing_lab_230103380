"""Phase 1: false-sharing stride sweep + process scaling with Karp-Flatt diagnostics.

Usage: python task1_falsesharing.py            (needs task0_cal.py to have been run first)
"""
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import (run_benchmark, emit_verify_tag, save_section, get_params,
                     seed_value, QUICK, quick_banner)


def fs_worker(shm_name: str, slot_idx: int, iters: int, nslots: int = 512) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * nslots).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc  # Forces store to shared L1 cache line
    del arr
    shm.close()


def measure(p, stride, M, warm, runs):
    # 4096-byte block = 512 slots; enlarged only if p*stride would not fit
    nslots = max(512, (p - 1) * stride + 1)
    shm = shared_memory.SharedMemory(create=True, size=nslots * 8)
    try:
        def fn():
            procs = [mp.Process(target=fs_worker, args=(shm.name, w * stride, M // p, nslots))
                     for w in range(p)]
            for pr in procs:
                pr.start()
            for pr in procs:
                pr.join()
        return run_benchmark(fn, warm, runs)
    finally:
        shm.close()
        shm.unlink()


def karp_flatt(S, p):
    return (1.0 / S - 1.0 / p) / (1.0 - 1.0 / p)


def main():
    quick_banner()
    pphys, plog = get_params()
    M = 1_600_000 if QUICK else 16_000_000
    warm, runs = (1, 3) if QUICK else (2, 7)
    print(f"P_phys={pphys}  P_log={plog}  M={M}")

    # ---- 1. stride sweep at p = Pphys --------------------------------------
    strides = [1, 2, 4, 8, 16, 32]
    sweep = []
    print(f"\n== Stride sweep (p={pphys}) ==")
    print(f"{'stride':>6} {'bytes':>6} {'min ms':>10} {'median ms':>10} {'IQR ms':>9} {'Mops/s':>9}")
    for s in strides:
        st = measure(pphys, s, M, warm, runs)
        mops = ((M // pphys) * pphys) / (st.median_ms / 1e3) / 1e6
        sweep.append({"stride": s, "bytes": s * 8, "min": st.min_ms, "median": st.median_ms,
                      "iqr": st.iqr_ms, "mops": mops})
        print(f"{s:>6} {s * 8:>6} {st.min_ms:>10.1f} {st.median_ms:>10.1f} {st.iqr_ms:>9.1f} {mops:>9.2f}")

    ratios = [(sweep[i]["median"] / sweep[i + 1]["median"], sweep[i + 1]) for i in range(len(sweep) - 1)]
    best_ratio, knee = max(ratios, key=lambda x: x[0])
    print(f"-> sharpest drop at stride={knee['stride']} (byte offset {knee['bytes']} B), "
          f"time falls x{best_ratio:.2f} versus the previous stride")

    # ---- 2. process scaling -------------------------------------------------
    ps = sorted({p for p in (1, 2, 4, 8, 16, 32, 64) if p <= plog} | {pphys, plog})
    scaling = {}
    for s in (1, 16):
        rows, t1 = [], None
        print(f"\n== Scaling, stride={s} ({s * 8} B) ==")
        print(f"{'p':>3} {'median ms':>10} {'IQR':>8} {'S_p':>6} {'E_p':>6} {'e(p)':>7}")
        for p in ps:
            st = measure(p, s, M, warm, runs)
            if p == 1:
                t1 = st.median_ms
            S = t1 / st.median_ms
            E = S / p
            e = karp_flatt(S, p) if p >= 2 else None
            rows.append({"p": p, "median": st.median_ms, "iqr": st.iqr_ms, "S": S, "E": E, "e": e})
            print(f"{p:>3} {st.median_ms:>10.1f} {st.iqr_ms:>8.1f} {S:>6.2f} {E:>6.2f} "
                  f"{'-' if e is None else f'{e:.3f}':>7}")
        scaling[str(s)] = rows

    m1 = next(r["median"] for r in sweep if r["stride"] == 1)
    m16 = next(r["median"] for r in sweep if r["stride"] == 16)
    tag = emit_verify_tag("PHASE1", seed_value(), m1, m16)
    print("\n" + tag)
    save_section("phase1", {"M": M, "pphys": pphys, "sweep": sweep, "scaling": scaling,
                            "knee_stride": knee["stride"], "knee_bytes": knee["bytes"],
                            "knee_ratio": best_ratio, "hash": tag, "quick": QUICK})


if __name__ == "__main__":
    main()

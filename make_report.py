"""make_report.py - builds lab_report_<ID>.md from lab_results.json (run AFTER task0..task4).

Usage: python make_report.py --id=<YourStudentID>
"""
import sys
from harness import load_results, get_student_id

FILES = ["harness.py", "task0_cal.py", "task1_falsesharing.py", "task2_scaling.py",
         "task3_roofline_gemm.py", "task4_scheduling.py"]


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    R = load_results()
    missing = [k for k in ("phase0", "phase1", "phase2", "phase3", "phase4") if k not in R]
    if missing:
        sys.exit(f"Missing results for {missing}. Run the task scripts first.")
    sid = get_student_id()
    p0, p1, p2, p3, p4 = (R[k] for k in ("phase0", "phase1", "phase2", "phase3", "phase4"))
    hw = p0["hw"]
    kb = lambda v: f"{v / 1024:.0f} KB" if v else "n/a"
    pphys, plog = hw["pphys"], hw["plog"]
    o = []
    w = o.append

    w(f"# Parallel Computing Python Lab - Report (ID: {sid})\n")
    if any(R[k].get("quick") for k in ("phase1", "phase2", "phase3", "phase4")):
        w("> WARNING: some results come from `--quick` mode. Re-run without `--quick` before submitting.\n")

    # ---------- 1. hardware + hashes
    w("## 1. Phase 0 - Hardware table\n")
    w("| Parameter | Value |\n|---|---|")
    w(f"| CPU model | {hw['model']} |\n| Physical cores (P_phys) | {pphys} |\n| Logical threads (P_log) | {plog} |")
    w(f"| Cache line (L_bytes) | {hw['line']} B |\n| L1d | {kb(hw['l1d'])} |\n| L2 | {kb(hw['l2'])} |\n| L3 | {kb(hw['l3'])} |")
    w(f"| perf_counter_ns latency | {p0['ns_per_call']:.1f} ns/call (avg of 10^6 calls) |\n")
    w("### Verification hashes\n\n```")
    for k in ("phase0", "phase1", "phase2", "phase3", "phase4"):
        w(R[k]["hash"])
    w("```\n")

    # ---------- 2. tables
    w("## 2. Measured results\n")
    w(f"### Phase 1 - Stride sweep (p = {pphys}, M = {p1['M']:,})\n")
    w("| stride | bytes | min ms | median ms | IQR ms | Mops/s |\n|---:|---:|---:|---:|---:|---:|")
    for r in p1["sweep"]:
        w(f"| {r['stride']} | {r['bytes']} | {f(r['min'], 1)} | {f(r['median'], 1)} | {f(r['iqr'], 1)} | {f(r['mops'])} |")
    w(f"\nSharpest drop at **stride = {p1['knee_stride']} ({p1['knee_bytes']} B)**, time falls x{p1['knee_ratio']:.2f}.\n")
    for s in ("1", "16"):
        w(f"### Phase 1 - Scaling, stride = {s} ({int(s) * 8} B)\n")
        w("| p | median ms | IQR | S_p | E_p | Karp-Flatt e(p) |\n|---:|---:|---:|---:|---:|---:|")
        for r in p1["scaling"][s]:
            w(f"| {r['p']} | {f(r['median'], 1)} | {f(r['iqr'], 1)} | {f(r['S'])} | {f(r['E'])} | {f(r['e'], 3)} |")
        w("")

    w(f"### Phase 2 - Strong scaling (N = {p2['N']:,})\n")
    w("| p | T_p ms | IQR | S_p | E_p | T_StageA ms |\n|---:|---:|---:|---:|---:|---:|")
    for r in p2["strong"]:
        w(f"| {r['p']} | {f(r['T'], 1)} | {f(r['iqr'], 1)} | {f(r['S'])} | {f(r['E'])} | {f(r['tA'], 1)} |")
    w(f"\ns = T_A(1)/T_1 = {p2['s']:.4f}, Amdahl cap S_inf = {p2['S_inf']:.1f}; "
      f"observed S({plog}) = {p2['strong'][-1]['S']:.2f}.\n")
    w(f"### Phase 2 - Weak scaling (N0 = {p2['N0']:,})\n")
    w("| p | N | T_p ms | IQR | E_weak | alpha | S_Gustafson |\n|---:|---:|---:|---:|---:|---:|---:|")
    for r in p2["weak"]:
        w(f"| {r['p']} | {r['N']:,} | {f(r['T'], 1)} | {f(r['iqr'], 1)} | {f(r['Ew'])} | {f(r['alpha'], 3)} | {f(r['SG'])} |")
    w("")

    w(f"### Phase 3 - Roofline and GEMM (N = {p3['N']}, p = {pphys})\n")
    w(f"- BW_sustained = **{p3['bw_gbs']:.2f} GB/s**\n- P_sustained = **{p3['p_sustained']:.3f} GFLOP/s**"
      f"\n- Ridge point I* = **{p3['ridge']:.4f} FLOP/B**\n")
    w("| Variant | median ms | GFLOP/s |\n|---|---:|---:|")
    w(f"| V1 naive i-j-k | {f(p3['v1']['ms'], 1)} | {f(p3['v1']['gflops'], 4)} |")
    w(f"| V2 reordered i-k-j | {f(p3['v2']['ms'], 1)} | {f(p3['v2']['gflops'], 4)} |")
    w("\n| BS | median ms | IQR | GFLOP/s | W_tile KB | I_tiled = BS/12 | fits in |\n|---:|---:|---:|---:|---:|---:|---|")
    for t in p3["tiles"]:
        w(f"| {t['bs']} | {f(t['ms'], 1)} | {f(t['iqr'], 1)} | {f(t['gflops'], 4)} | {t['wtile'] / 1024:.1f} | {t['I']:.2f} | {t['fits']} |")
    w(f"\nBS* = **{p3['bs_star']}** ({p3['bs_star_gflops']:.4f} GFLOP/s). Verification max|C3-C2| = {p3['max_err']:.2e} "
      f"({'OK' if p3['verified'] else 'FAILED'}, threshold 1e-9).\n")

    w(f"### Phase 4 - Scheduling (N = {p4['N']:,}, p = {p4['p']})\n")
    w("| Strategy | T_wall ms | IQR | lambda |\n|---|---:|---:|---:|")
    w(f"| Block static | {f(p4['block']['T'], 1)} | {f(p4['block']['iqr'], 1)} | {f(p4['block']['lam'], 3)} |")
    w(f"| Cyclic static | {f(p4['cyclic']['T'], 1)} | {f(p4['cyclic']['iqr'], 1)} | {f(p4['cyclic']['lam'], 3)} |")
    for d in p4["dynamic"]:
        w(f"| Dynamic C={d['C']} | {f(d['T'], 1)} | {f(d['iqr'], 1)} | {f(d['lam'], 3)} |")
    w(f"\nC* = **{p4['c_star']}**.\n")

    # ---------- 3. analysis (data-driven text)
    w("## 3. Diagnostic answers\n")
    e1 = [r["e"] for r in p1["scaling"]["1"] if r["e"] is not None]
    e16 = [r["e"] for r in p1["scaling"]["16"] if r["e"] is not None]
    if p1["knee_ratio"] >= 1.15:
        knee_txt = (f"Throughput rises once the slots are {p1['knee_bytes']} B apart (stride {p1['knee_stride']}), "
                    f"i.e. when each worker owns a private line (L_bytes = {hw['line']} B). ")
    else:
        knee_txt = (f"No sharp drop in contention was observed: the largest step is only x{p1['knee_ratio']:.2f} "
                    f"(stride {p1['knee_stride']}), comparable to the IQR/noise. The expected boundary is the cache line "
                    f"L_bytes = {hw['line']} B (stride {hw['line'] // 8}); the pure-Python loop (~100+ ns per iteration) is too slow "
                    "relative to coherence latency, so the interpreter hides most of the false-sharing penalty. ")
    t = ("**Phase 1.** With stride = 1 all workers write 8-byte slots inside the same cache line, so each store "
         "invalidates the line in the other cores' L1 (coherence ping-pong). " + knee_txt)
    if e1 and e16:
        t += (f"Karp-Flatt e(p) for stride = 1 ranges {min(e1):.2f}-{max(e1):.2f} and is expected to grow with p "
              "(coherence traffic is overhead that scales with the number of contenders), whereas for stride = 16 it is "
              f"lower/flatter ({min(e16):.2f}-{max(e16):.2f}): the remaining serial fraction is process start-up and "
              "memory-system limits. (Compare with your table; CPython overhead can mask part of the contention.)")
    w(t + "\n")

    inf = p2["inflection"]
    w(f"**Phase 2.** Serial Stage A is s = {p2['s']:.3f} of T_1, so Amdahl caps speedup at {p2['S_inf']:.1f}; "
      f"the observed S({plog}) = {p2['strong'][-1]['S']:.2f} is far below that cap, therefore on this machine the limit is the number of cores "
      "(and memory/SMT effects), not the serial fraction. Weak scaling keeps per-worker load constant: E_weak shows how much "
      "the growing serial Stage A (0.05*N) and Pool overhead cost, and Gustafson's S = p - alpha(p-1) "
      "is higher than strong speedup because the problem grows with p.")
    if inf:
        bd = inf["boundary"]
        w(f" Boundary ({inf['kind']}) at p = {bd}: marginal speedup p={bd - 1 if bd > 1 else 1}->{bd} is {inf['dS_before']:+.2f}, while "
          f"p={bd}->{bd + 1} is {inf['dS_after']:+.2f} (a {inf['drop_pct']:.0f}% drop). Beyond the boundary the extra workers land on "
          "SMT siblings / slower efficiency cores, and with equal static chunks the slowest core decides the total time.\n")
    else:
        w(f" P_log = P_phys = {pphys} and no hybrid cores detected, so there is no boundary to quantify.\n")

    ai_v = p3["ridge"]
    w(f"**Phase 3.** Ridge point I* = {p3['ridge']:.4f} FLOP/B: kernels with intensity below I* are bandwidth-bound "
      f"(roof = BW*I), above it compute-bound (roof = {p3['p_sustained']:.3f} GFLOP/s). V1 (naive i-j-k) walks B with stride N*8 B = "
      f"{p3['N'] * 8} B, a new cache line per access, so it has the worst locality; V2 (i-k-j) makes the inner loop unit-stride "
      f"(x{p3['v2']['gflops'] / p3['v1']['gflops']:.2f} vs V1). V3 tiles so that 3*BS^2*8 B stays in cache; the best tile "
      f"BS* = {p3['bs_star']} needs W_tile = {3 * p3['bs_star'] ** 2 * 8 / 1024:.1f} KB against L1d = {kb(hw['l1d'])} and L2 = {kb(hw['l2'])}. "
      f"Its intensity I = BS*/12 = {p3['bs_star'] / 12:.2f} FLOP/B is {'above' if p3['bs_star'] / 12 > ai_v else 'below'} the ridge point, "
      "so on the roofline it sits "
      f"{'under the compute roof' if p3['bs_star'] / 12 > ai_v else 'under the bandwidth slope'}. Pure-Python GEMM runs far below "
      "the hardware roof (interpreter overhead dominates: each FLOP costs many bytecodes), which is why all three variants lie far "
      "under the roofline; blocking helps only to the extent that memory latency is visible through the interpreter overhead. "
      "Too small a tile (8) loses to loop overhead, too large (>= the L1/L2 limit) spills out of cache.\n")

    w(f"**Phase 4.** Item cost grows quadratically with i, so Block static gives the last worker almost all the work "
      f"(lambda = {p4['block']['lam']:.2f}), Cyclic static interleaves cheap and expensive items (lambda = {p4['cyclic']['lam']:.2f}), "
      f"and the dynamic queue balances by construction. C = 1 gives the best balance but each item takes the shared lock "
      f"(synchronization contention, lambda = {p4['dynamic'][0]['lam']:.2f}, T = {p4['dynamic'][0]['T']:.0f} ms); "
      f"C = 512 has almost no lock traffic but the last chunks (the heaviest, at the end of the range) create a tail "
      f"imbalance (lambda = {p4['dynamic'][-1]['lam']:.2f}, T = {p4['dynamic'][-1]['T']:.0f} ms). "
      f"The best trade-off is C* = {p4['c_star']} (T = "
      f"{min(d['T'] for d in p4['dynamic']):.0f} ms, x{p4['block']['T'] / min(d['T'] for d in p4['dynamic']):.2f} faster than Block static).\n")

    # ---------- 4. code
    w("## 4. Source code\n")
    for fn in FILES + ["make_report.py"]:
        w(f"### {fn}\n\n```python")
        with open(fn, encoding="utf-8") as fh:
            w(fh.read().rstrip())
        w("```\n")

    out = f"lab_report_{sid}.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o))
    print("written", out)


if __name__ == "__main__":
    main()

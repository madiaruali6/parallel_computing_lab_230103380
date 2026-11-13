"""Phase 0 checkpoint: cpu_count, perf_counter_ns call latency, hardware table, VERIFY_HASH[PHASE0].

Usage:  python task0_cal.py --id=<YourStudentID>
Missing cache values can be forced with env vars, e.g.  L1D_KB=48 L2_KB=1280 L3_KB=12288 LINE=64 PPHYS=6
"""
import time
import sys
import multiprocessing as mp
from harness import (emit_verify_tag, detect_hw, save_section, get_student_id,
                     seed_value, quick_banner)


def main():
    quick_banner()
    if not any(a.startswith("--id=") for a in sys.argv[1:]):
        try:
            sid = input("Student ID: ").strip() or "0"
            sys.argv.append(f"--id={sid}")
        except EOFError:
            pass
    sid = get_student_id()
    plog = mp.cpu_count()

    n = 10 ** 6
    pc = time.perf_counter_ns
    for _ in range(1000):          # warm-up
        pc()
    t0 = pc()
    for _ in range(n):
        pc()
    t1 = pc()
    avg_ns = (t1 - t0) / n

    hw = detect_hw()
    kb = lambda v: f"{v / 1024:.0f} KB" if v else "n/a (fill in manually)"
    print("=== Phase 0: hardware ===")
    print(f"CPU model        : {hw['model']}")
    print(f"Physical cores   : {hw['pphys']}")
    print(f"Logical threads  : {plog}")
    print(f"Cache line       : {hw['line']} B")
    print(f"L1d / L2 / L3    : {kb(hw['l1d'])} / {kb(hw['l2'])} / {kb(hw['l3'])}")
    print(f"perf_counter_ns  : {avg_ns:.1f} ns per call (avg over {n} calls)")

    tag = emit_verify_tag("PHASE0", seed_value(), avg_ns, float(plog))
    print(tag)
    save_section("phase0", {"student_id": sid, "hw": hw, "ns_per_call": avg_ns, "hash": tag})


if __name__ == "__main__":
    main()

"""harness.py - Minimal timing / verification harness + small helpers shared by all tasks."""
from dataclasses import dataclass
import multiprocessing as mp
import time
import os
import sys
import json
import re
import subprocess
from typing import Callable

RESULTS_FILE = "lab_results.json"
QUICK = "--quick" in sys.argv   # tiny workloads for a smoke test ONLY (hashes are NOT valid)


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup: int = 2, runs: int = 7) -> BenchStats:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    med = samples[runs // 2]
    return BenchStats(samples[0], med, samples[(3 * runs) // 4] - samples[runs // 4])


def emit_verify_tag(phase: str, seed: int, m1: float, m2: float) -> str:
    hw = mp.cpu_count()
    sig = fnv1a_hash(f"{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH[{phase}]: {sig:016X} (P_log={hw})"


# ----------------------------------------------------------------------------
# helpers (not part of the original listing)
# ----------------------------------------------------------------------------
def stats_from(samples_ms) -> BenchStats:
    s = sorted(samples_ms)
    n = len(s)
    return BenchStats(s[0], s[n // 2], s[(3 * n) // 4] - s[n // 4])


def load_results() -> dict:
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_section(key: str, data: dict) -> None:
    r = load_results()
    r[key] = data
    with open(RESULTS_FILE, "w") as f:
        json.dump(r, f, indent=2)


def get_student_id() -> str:
    for a in sys.argv[1:]:
        if a.startswith("--id="):
            return a[5:]
    sid = os.environ.get("STUDENT_ID")
    if sid:
        return sid
    return str(load_results().get("phase0", {}).get("student_id", "0"))


def seed_value():
    sid = get_student_id()
    return int(sid) if sid.isdigit() else sid


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def detect_pphys() -> int:
    env = os.environ.get("PPHYS")
    if env:
        return int(env)
    plog = mp.cpu_count()
    n = None
    try:
        if sys.platform.startswith("linux"):
            out = _run(["lscpu", "-p=CORE,SOCKET"])
            pairs = {l for l in out.splitlines() if l and not l.startswith("#")}
            n = len(pairs) or None
        elif sys.platform == "darwin":
            n = int(_run(["sysctl", "-n", "hw.physicalcpu"]))
        elif sys.platform.startswith("win"):
            n = int(_run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"]))
    except Exception:
        n = None
    if not n or n < 1 or n > plog:
        print("WARNING: could not detect physical cores, using P_log. Set env PPHYS=<n> to override.")
        n = plog
    return n


def _parse_size(s: str):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?)", s.upper())
    if not m:
        return None
    mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
    return int(float(m.group(1)) * mult)


def detect_hw() -> dict:
    """Best effort. Anything missing can be given via env: CPU_MODEL, L1D_KB, L2_KB, L3_KB, LINE."""
    hw = {"model": None, "pphys": detect_pphys(), "plog": mp.cpu_count(),
          "line": None, "l1d": None, "l2": None, "l3": None}
    try:
        if sys.platform.startswith("linux"):
            for l in _run(["lscpu"]).splitlines():
                if l.startswith("Model name"):
                    hw["model"] = l.split(":", 1)[1].strip()
            base = "/sys/devices/system/cpu/cpu0/cache"
            if os.path.isdir(base):
                for d in sorted(os.listdir(base)):
                    if not d.startswith("index"):
                        continue
                    rd = lambda n: open(os.path.join(base, d, n)).read().strip()
                    try:
                        lvl, typ, size = rd("level"), rd("type"), _parse_size(rd("size"))
                        if lvl == "1" and typ in ("Data", "Unified"):
                            hw["l1d"] = size
                        elif lvl == "2":
                            hw["l2"] = size
                        elif lvl == "3":
                            hw["l3"] = size
                        hw["line"] = int(rd("coherency_line_size"))
                    except Exception:
                        pass
            if not hw["line"]:
                v = _run(["getconf", "LEVEL1_DCACHE_LINESIZE"])
                hw["line"] = int(v) if v.isdigit() and int(v) > 0 else None
        elif sys.platform == "darwin":
            hw["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            g = lambda k: (lambda v: int(v) if v.isdigit() else None)(_run(["sysctl", "-n", k]))
            hw["line"], hw["l1d"], hw["l2"], hw["l3"] = (g("hw.cachelinesize"), g("hw.l1dcachesize"),
                                                         g("hw.l2cachesize"), g("hw.l3cachesize"))
        elif sys.platform.startswith("win"):
            q = lambda prop: _run(["powershell", "-NoProfile", "-Command",
                                   f"(Get-CimInstance Win32_Processor | Select-Object -First 1).{prop}"])
            hw["model"] = q("Name")
            l2, l3 = q("L2CacheSize"), q("L3CacheSize")
            hw["l2"] = int(l2) * 1024 if l2.isdigit() else None
            hw["l3"] = int(l3) * 1024 if l3.isdigit() else None
    except Exception:
        pass
    # manual overrides
    if os.environ.get("CPU_MODEL"):
        hw["model"] = os.environ["CPU_MODEL"]
    for key, env in (("l1d", "L1D_KB"), ("l2", "L2_KB"), ("l3", "L3_KB")):
        if os.environ.get(env):
            hw[key] = int(float(os.environ[env]) * 1024)
    if os.environ.get("LINE"):
        hw["line"] = int(os.environ["LINE"])
    hw["model"] = hw["model"] or "unknown"
    hw["line"] = hw["line"] or 64
    return hw


def detect_pcores():
    """Performance cores on hybrid CPUs (Apple Silicon). env PCORES overrides. None if not hybrid/unknown."""
    if os.environ.get("PCORES"):
        return int(os.environ["PCORES"])
    if sys.platform == "darwin":
        v = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if v.isdigit():
            return int(v)
    return None


def get_hw() -> dict:
    p0 = load_results().get("phase0")
    return p0["hw"] if p0 else detect_hw()


def get_params():
    hw = get_hw()
    return hw["pphys"], hw["plog"]


def quick_banner():
    if QUICK:
        print("*** QUICK MODE: tiny workloads, results/hash are NOT valid for submission ***")

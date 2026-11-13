"""fix_cache.py - correct cache sizes in lab_results.json to the P-core values (Apple Silicon), without re-running benchmarks.
Usage: python3 fix_cache.py        then:  python3 make_report.py --id=<ID>
Hashes are NOT affected (they depend only on timings and P_log)."""
import json, subprocess

def sc(k):
    try:
        v = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        return int(v) if v.isdigit() and int(v) > 0 else None
    except Exception:
        return None

R = json.load(open("lab_results.json"))
hw = R["phase0"]["hw"]
l1, l2 = sc("hw.perflevel0.l1dcachesize"), sc("hw.perflevel0.l2cachesize")
print("P-core L1d/L2 from sysctl:", l1, l2)
hw["l1d"] = l1 or hw["l1d"]
hw["l2"] = l2 or hw["l2"]
for t in R["phase3"]["tiles"]:
    w = t["wtile"]
    t["fits"] = "L1d" if hw["l1d"] and w <= hw["l1d"] else "L2" if hw["l2"] and w <= hw["l2"] else "?"
json.dump(R, open("lab_results.json", "w"), indent=2)
print("updated: L1d =", hw["l1d"], "L2 =", hw["l2"])

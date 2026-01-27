#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict

from problem import SLOT_LIMITS


def load_trace(path: str):
    with open(path, "r") as f:
        data = f.read()
    # The trace writer leaves a trailing comma before the closing bracket.
    data = re.sub(r",\s*]", "]", data.strip())
    return json.loads(data)


def analyze(events):
    tid_to_engine = {}
    engine_tids = set()
    for ev in events:
        if ev.get("ph") != "M":
            continue
        args = ev.get("args") or {}
        name = args.get("name", "")
        if "-" not in name:
            continue
        engine = name.split("-", 1)[0]
        if engine in SLOT_LIMITS:
            tid = ev.get("tid")
            if tid is not None:
                tid_to_engine[tid] = engine
                engine_tids.add(tid)

    engine_counts = defaultdict(int)
    engine_cycles = defaultdict(set)
    total_instructions = 0
    max_ts = -1

    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "op":
            continue
        if ev.get("name") == "init":
            continue
        tid = ev.get("tid")
        engine = tid_to_engine.get(tid)
        if engine is None:
            continue
        ts = ev.get("ts")
        if ts is None:
            continue
        engine_counts[engine] += 1
        engine_cycles[engine].add(ts)
        total_instructions += 1
        if ts > max_ts:
            max_ts = ts

    cycles = max_ts + 1 if max_ts >= 0 else 0
    return {
        "cycles": cycles,
        "engine_counts": dict(engine_counts),
        "engine_cycles": {k: len(v) for k, v in engine_cycles.items()},
        "total_instructions": total_instructions,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Perfetto trace.json")
    parser.add_argument("trace", nargs="?", default="trace.json")
    args = parser.parse_args()

    events = load_trace(args.trace)
    stats = analyze(events)

    cycles = stats["cycles"]
    total_insts = stats["total_instructions"] or 1

    print(f"Trace: {args.trace}")
    print(f"Cycles: {cycles}")
    print("")
    print("Hardware Utilization:")
    for engine, limit in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        used = stats["engine_counts"].get(engine, 0)
        capacity = cycles * limit
        util = (used / capacity * 100) if capacity else 0.0
        active_cycles = stats["engine_cycles"].get(engine, 0)
        active_util = (used / (active_cycles * limit) * 100) if active_cycles else 0.0
        status = "SATURATED" if util > 95 else "High Load" if util > 75 else "Underutilized"
        print(
            f"{engine.upper():5s}: Capacity {limit:2d} slots/cycle | Utilization {util:6.2f}% (Active {active_util:6.2f}%) | Status: {status}"
        )

    print("")
    print("Instruction Mix:")
    print(f"Total Instructions: {stats['total_instructions']}")
    for engine in ["valu", "alu", "load", "store", "flow"]:
        count = stats["engine_counts"].get(engine, 0)
        pct = count / total_insts * 100
        print(f"{engine.upper():5s}: {count:5d} ({pct:5.1f}%)")
    ipc = total_insts / cycles if cycles else 0.0
    print(f"Instructions Per Cycle: {ipc:.2f}")

    print("")
    print("Elite Diagnostics:")
    load_ops = stats["engine_counts"].get("load", 0)
    valu_ops = stats["engine_counts"].get("valu", 0)
    store_ops = stats["engine_counts"].get("store", 0)
    
    load_pressure = load_ops / cycles if cycles else 0
    compute_ratio = valu_ops / (load_ops + store_ops) if (load_ops + store_ops) else 0
    # Total capacity = sum(SLOT_LIMITS[engine] for engine in SLOT_LIMITS if engine != "debug")
    # alu:12, valu:6, load:2, store:2, flow:1 -> 23 slots
    packing_density = total_insts / (cycles * 23) if cycles else 0

    print(f"Memory Pressure:  {load_pressure:.2f} / 2.00 (Loads per Cycle)")
    print(f"Compute Ratio:    {compute_ratio:.2f} (VALU Ops per Memory Op)")
    print(f"Packing Density:  {packing_density*100:.1f}% of total hardware capacity used")

if __name__ == "__main__":
    main()

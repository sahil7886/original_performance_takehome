
import json
import re
import math
import argparse
from collections import defaultdict, Counter
import sys

# Constants from problem.py
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}
VLEN = 8
ENGINES = ["alu", "valu", "load", "store", "flow"]

def load_trace(path):
    print(f"Loading trace from {path}...")
    with open(path, "r") as f:
        data = f.read()
    # Fix trailing comma
    data = re.sub(r",\s*]", "]", data.strip())
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

def parse_slot_string(slot_str):
    # slot_str is like "('+', 8, 4, 0)" or "('vbroadcast', 32, 16)"
    try:
        return eval(slot_str)
    except:
        return None

def get_io(slot):
    if not isinstance(slot, tuple):
        return set(), set()
    
    op = slot[0]
    reads = set()
    writes = set()
    
    args = slot[1:]
    
    if op in ["store", "vstore"]:
        for a in args:
            reads.add(a)
    elif op in ["cond_jump", "cond_jump_rel", "jump_indirect", "trace_write"]:
        for a in args:
            reads.add(a)
    elif op in ["jump", "halt", "pause"]:
        pass
    elif op == "compare": # debug
        for a in args:
            reads.add(a)
    else:
        if len(args) > 0:
            writes.add(args[0])
            for a in args[1:]:
                reads.add(a)
                
    return reads, writes

def analyze_metrics(events):
    print("Analyzing metrics...")
    
    cycles = defaultdict(list)
    scratch_writes = defaultdict(dict)
    
    max_ts = 0
    tid_to_engine = {}
    
    # Pre-scan for TIDs
    for ev in events:
        if ev["ph"] == "M" and "name" in ev.get("args", {}):
            name = ev["args"]["name"]
            if "-" in name: 
                part = name.split("-")[0]
                if part in SLOT_LIMITS:
                    tid_to_engine[ev["tid"]] = part
    
    # Parsing ops
    for ev in events:
        ts = ev.get("ts", 0)
        max_ts = max(max_ts, ts)
        
        if ev["ph"] == "X" and ev.get("cat") == "op":
            name = ev["name"]
            tid = ev.get("tid")
            
            if tid is not None and tid >= 100000:
                addr = tid - 100000
                val_str = name
                try:
                    if "," in val_str:
                        val = [int(x.strip()) for x in val_str.split(",") if x.strip()]
                    else:
                        val = [int(val_str.strip())] if val_str.strip() else []
                    scratch_writes[ts][addr] = val
                except:
                    pass
            elif tid in tid_to_engine:
                engine = tid_to_engine[tid]
                if name == "init": continue
                
                slot_str = ev["args"].get("slot", "()")
                slot = parse_slot_string(slot_str)
                
                cycles[ts].append({
                    "engine": engine,
                    "slot": slot,
                    "name": name
                })

    total_cycles = max_ts + 1
    
    # --- Metrics State ---
    
    # 1. Vertical & 2. Horizontal
    filled_cycles = len(cycles)
    vertical_waste = (total_cycles - filled_cycles) / total_cycles if total_cycles > 0 else 0
    bundle_sizes = []
    
    # 3. Saturation
    engine_saturation = defaultdict(int)
    
    # 4. Diversity
    bundle_diversity = []
    
    # 5. Raw Hazards
    last_write_cycle = {} 
    raw_hazard_count = 0
    
    # 6. Reuse Dist
    last_access_cycle = {}
    reuse_distances = []
    
    # 7. PC Entropy
    instruction_counts = Counter()
    
    # 8. Vector
    vector_ops_count = 0
    vector_distinct_lanes_count = 0
    
    # 9. Latency
    trace_write_times = []

    # --- NEW METRICS STATE ---
    
    # 10. Co-activation Matrix
    # Counts for pairs of engines active in the same cycle
    co_activation = defaultdict(int) 
    
    # 11. Instruction Reuse Distance (II)
    # Map static_op_signature -> last_ts
    inst_last_ts = {}
    inst_reuse_dists = []
    
    # 12. Working Set Pressure
    # Determine touched addresses per cycle first
    cycle_touched_addrs = defaultdict(set)
    SCRATCH_CAPACITY = 1536

    for ts in range(total_cycles):
        ops = cycles.get(ts, [])
        bundle_sizes.append(len(ops))
        
        # Engines active this cycle
        active_engines = set(op['engine'] for op in ops)
        
        # Saturation & Diversity
        eng_counts = Counter([op['engine'] for op in ops])
        for eng, limit in SLOT_LIMITS.items():
            if eng_counts[eng] == limit:
                engine_saturation[eng] += 1
        
        if len(ops) > 0:
            entropy = 0
            for eng, count in eng_counts.items():
                p = count / len(ops)
                entropy -= p * math.log2(p)
            bundle_diversity.append(entropy)
        else:
            bundle_diversity.append(0)
            
        # Co-activation
        # We want pairs of active engines.
        # Since we want a matrix, we iterate all standard engines
        uni_engines = sorted(list(active_engines))
        for i in range(len(uni_engines)):
            for j in range(i, len(uni_engines)):
                e1 = uni_engines[i]
                e2 = uni_engines[j]
                # Increment pair (symmetric)
                co_activation[(e1, e2)] += 1
                if e1 != e2:
                    co_activation[(e2, e1)] += 1
        
        current_reads = set()
        current_writes = set()
        
        for op_info in ops:
            slot = op_info['slot']
            if not slot: continue
            
            # Instruction Reuse (II)
            # Signature: Use the full tuple string as the 'static' op
            sig = str(slot)
            if sig in inst_last_ts:
                dist = ts - inst_last_ts[sig]
                inst_reuse_dists.append(dist)
            inst_last_ts[sig] = ts
            
            instruction_counts[sig] += 1
            
            r, w = get_io(slot)
            
            # Accumulate touched addresses for Working Set
            for addr in r:
                if isinstance(addr, int):
                    cycle_touched_addrs[ts].add(addr)
                    # Reuse dist logic (general addr reuse)
                    if addr in last_access_cycle:
                        reuse_distances.append(ts - last_access_cycle[addr])
                    last_access_cycle[addr] = ts
                    current_reads.add(addr)
            
            for addr in w:
                if isinstance(addr, int):
                    cycle_touched_addrs[ts].add(addr)
                    # Reuse dist logic
                    if addr in last_access_cycle:
                        reuse_distances.append(ts - last_access_cycle[addr])
                    last_access_cycle[addr] = ts
                    current_writes.add(addr)
            
            # Vector logic
            engine = op_info['engine']
            op_name = op_info['name']
            if engine == "valu" or op_name.startswith("v"):
                 if w:
                     dest = next(iter(w))
                     # Check actual values logged
                     val = scratch_writes.get(ts, {}).get(dest)
                     if val and len(val) > 1:
                         vector_ops_count += 1
                         if len(set(val)) > 1:
                             vector_distinct_lanes_count += 1
            
            # Latency
            if op_name == "trace_write":
                trace_write_times.append(ts)
        
        # RAW Hazards
        for r_addr in current_reads:
            if isinstance(r_addr, int):
                # Hazard if written in previous cycle
                if last_write_cycle.get(r_addr) == ts - 1:
                    raw_hazard_count += 1
        
        for w_addr in current_writes:
            if isinstance(w_addr, int):
                last_write_cycle[w_addr] = ts

    # Post-Calculation
    
    # PC Entropy
    total_insts = sum(instruction_counts.values())
    pc_entropy = 0
    if total_insts > 0:
        for count in instruction_counts.values():
            p = count / total_insts
            pc_entropy -= p * math.log2(p)
            
    # Latency
    traversal_latencies = []
    if len(trace_write_times) >= 2:
        for i in range(1, len(trace_write_times)):
            traversal_latencies.append(trace_write_times[i] - trace_write_times[i-1])
    avg_latency = sum(traversal_latencies) / len(traversal_latencies) if traversal_latencies else 0

    # Working Set Pressure (Sliding Window N=50)
    WINDOW = 50
    ws_pressures = []
    # Optimization: Reconstruct full set isn't efficient for huge traces, but OK for takehome size (e.g. 50k cycles)
    # We can do a sliding window using a counter
    
    current_window_counts = Counter()
    # Init first window
    for t in range(min(WINDOW, total_cycles)):
        for addr in cycle_touched_addrs[t]:
            current_window_counts[addr] += 1
            
    ws_pressures.append(len(current_window_counts))
    
    for t in range(WINDOW, total_cycles):
        # Remove t-WINDOW
        old_t = t - WINDOW
        for addr in cycle_touched_addrs[old_t]:
            current_window_counts[addr] -= 1
            if current_window_counts[addr] == 0:
                del current_window_counts[addr]
        
        # Add t
        for addr in cycle_touched_addrs[t]:
            current_window_counts[addr] += 1
            
        ws_pressures.append(len(current_window_counts))
        
    avg_ws = sum(ws_pressures) / len(ws_pressures) if ws_pressures else 0
    max_ws = max(ws_pressures) if ws_pressures else 0
    
    # II (Instruction Reuse)
    avg_ii = sum(inst_reuse_dists)/len(inst_reuse_dists) if inst_reuse_dists else 0

    print("-" * 60)
    print("KERNEL PERFORMANCE ACCOUNTING")
    print("-" * 60)
    print(f"1.  Vertical Waste:          {vertical_waste * 100:.2f}%")
    
    print("2.  Horizontal Waste (Histogram):")
    bins = {0:0, 1:0, 6:0, 11:0}
    for s in bundle_sizes:
        if s == 0: bins[0]+=1
        elif s <= 5: bins[1]+=1
        elif s <= 10: bins[6]+=1
        else: bins[11]+=1
    total = len(bundle_sizes)
    print(f"    0 slots:    {bins[0]:5d} ({bins[0]/total*100:.1f}%)")
    print(f"    1-5 slots:  {bins[1]:5d} ({bins[1]/total*100:.1f}%)")
    print(f"    6-10 slots: {bins[6]:5d} ({bins[6]/total*100:.1f}%)")
    print(f"    11+ slots:  {bins[11]:5d} ({bins[11]/total*100:.1f}%)")
    
    print("3.  Saturation Events:")
    for eng, count in engine_saturation.items():
        print(f"    {eng.upper():5s}: {count} cycles")
        
    avg_div = sum(bundle_diversity)/len(bundle_diversity) if bundle_diversity else 0
    print(f"4.  Bundle Diversity Index:  {avg_div:.2f} bits")
    
    print(f"5.  Scratchpad RAW Hazards:  {raw_hazard_count}")
    
    avg_reuse = sum(reuse_distances)/len(reuse_distances) if reuse_distances else 0
    print(f"6.  Avg Address Reuse Dist:  {avg_reuse:.2f} cycles")
    
    print(f"7.  PC Entropy:              {pc_entropy:.2f} bits")
    
    vec_util_distinct = vector_distinct_lanes_count / vector_ops_count * 100 if vector_ops_count else 0
    print(f"8.  Vector Lane Variance:    {vec_util_distinct:.1f}%")
    
    print(f"9.  Avg Latency per Node:    {avg_latency:.1f} cycles")
    
    print("10. Engine Co-activation Matrix:")
    # Header
    print("       " + " ".join([f"{e:>6s}" for e in ENGINES]))
    for e1 in ENGINES:
        row = [f"{co_activation[(e1, e2)]:6d}" for e2 in ENGINES]
        print(f"{e1:>6s} " + " ".join(row))
    print("    * Diagonal = Homogeneous bursts. Off-diagonal = Parallelism.")
        
    print(f"11. Instruction Reuse (II):  {avg_ii:.2f} cycles avg")

    print(f"12. Working Set (N={WINDOW}):   {avg_ws:.1f} unique addrs ({avg_ws/SCRATCH_CAPACITY*100:.1f}% cap)")
    print("-" * 60)


if __name__ == "__main__":
    trace_path = "trace.json"
    if len(sys.argv) > 1:
        trace_path = sys.argv[1]
    
    events = load_trace(trace_path)
    analyze_metrics(events)

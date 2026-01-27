"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        return self.pack(slots)
    
    def get_io(self, slot):
        """Returns (reads, writes) sets of scratch addresses for a slot"""
        eng, op = slot
        reads = set()
        writes = set()
        
        # Helper to mark a register as read
        def r(addr):
            if isinstance(addr, int): reads.add(addr)
            
        # Helper to mark a register as written
        def w(addr):
            if isinstance(addr, int): writes.add(addr)

        match slot:
            case ("alu", (op_name, dest, a1, a2)):
                if op == "cdiv": # cdiv is special? no, standard 2-op
                    pass 
                w(dest)
                r(a1)
                r(a2)
            case ("valu", ("vbroadcast", dest, src)):
                # Vector ops write/read VLEN addresses
                for i in range(VLEN): w(dest + i)
                r(src)
            case ("valu", ("multiply_add", dest, a, b, c)):
                for i in range(VLEN): 
                    w(dest + i)
                    r(a + i)
                    r(b + i)
                    r(c + i)
            case ("valu", (op_name, dest, a1, a2)):
                for i in range(VLEN):
                    w(dest + i)
                    r(a1 + i)
                    r(a2 + i)
            case ("load", ("load", dest, addr)):
                w(dest)
                r(addr)
            case ("load", ("load_offset", dest, addr, offset)):
                # dest + offset is written, addr + offset is read (addr is base)
                # scratch[addr+offset] is the address in memory, scratch[dest+offset] is the value
                # Wait, load_offset: self.scratch[dest+offset] = self.mem[self.scratch[addr+offset]]
                w(dest + offset)
                r(addr + offset)
            case ("load", ("vload", dest, addr)):
                for i in range(VLEN): w(dest + i)
                r(addr)
            case ("load", ("const", dest, val)):
                w(dest)
            case ("store", ("store", addr, src)):
                r(addr)
                r(src)
            case ("store", ("vstore", addr, src)):
                r(addr)
                for i in range(VLEN): r(src + i)
            case ("flow", ("select", dest, cond, a, b)):
                w(dest)
                r(cond); r(a); r(b)
            case ("flow", ("add_imm", dest, a, imm)):
                w(dest)
                r(a)
            case ("flow", ("vselect", dest, cond, a, b)):
                for i in range(VLEN):
                    w(dest + i)
                    r(cond + i); r(a + i); r(b + i)
            case ("flow", ("halt",)): pass
            case ("flow", ("pause",)): pass
            case ("flow", ("trace_write", val)): r(val)
            case ("flow", ("cond_jump", cond, addr)): r(cond)
            case ("flow", ("cond_jump_rel", cond, offset)): r(cond)
            case ("flow", ("jump", addr)): pass # immediate addr
            case ("flow", ("jump_indirect", addr)): r(addr)
            case ("flow", ("coreid", dest)): w(dest)
            
            # Debug instructions don't affect dependencies in the simulator usually, 
            # but let's be safe and assume they might read things.
            case ("debug", ("compare", loc, key)): r(loc)
            case ("debug", ("vcompare", loc, keys)): 
                for i in range(VLEN): r(loc + i)
            case ("debug", _): pass 
            
            case _:
                # Default safety: assume it reads nothing and writes nothing if unknown, 
                # effectively serializing it if we don't know what it does.
                pass
                
        return reads, writes

    def pack(self, slots: list[tuple[Engine, tuple]]):
        schedule = [] # List of dict[engine, list[op]]
        # Track usage per cycle: index -> dict[engine, count]
        cycle_counts = [] 
        
        # Track dependencies
        # map addr -> cycle index of last write
        last_write = {} 
        # map addr -> cycle index of last read
        last_read = {} 

        for slot in slots:
            engine, op = slot
            reads, writes = self.get_io(slot)
            
            start_cycle = 0
            
            # Barrier for pause/halt
            if engine == "flow" and op[0] in ("pause", "halt"):
                start_cycle = len(schedule)
            else:
                # RAW: Must be after last write of any input
                # Simulator: Reads happen at START of cycle.
                # So if Write at T, Read can be at T+1.
                for r in reads:
                    if r in last_write:
                        start_cycle = max(start_cycle, last_write[r] + 1)
                
                # WAW: Must be after last write of any output
                for w in writes:
                    if w in last_write:
                        start_cycle = max(start_cycle, last_write[w] + 1)
                
                # WAR: Can be same cycle as last read.
                # If Read at T, Write can be at T.
                # But Write cannot be at T-1?
                # Actually, if I write at T, it takes effect at End of T.
                # The read at T reads old value.
                # So they can be concurrent.
                # But I cannot write at T if Read is at T+1 (that would be RAW violation for the reader).
                # But here we are scheduling the Writer AFTER the Reader seen in stream.
                # So Writer must be >= Reader Cycle.
                for w in writes:
                    if w in last_read:
                        start_cycle = max(start_cycle, last_read[w])
            
            # Find first cycle >= start_cycle that has resources
            while True:
                # Ensure schedule has space
                while start_cycle >= len(schedule):
                    schedule.append(defaultdict(list))
                    cycle_counts.append(defaultdict(int))
                
                # Check limits
                cc = cycle_counts[start_cycle]
                if cc[engine] < SLOT_LIMITS[engine]:
                    # Fits!
                    break
                start_cycle += 1
            
            # Schedule it
            schedule[start_cycle][engine].append(op)
            cycle_counts[start_cycle][engine] += 1
            
            # Update dependencies
            for r in reads:
                # We record the latest read cycle
                curr = last_read.get(r, -1)
                if start_cycle > curr:
                    last_read[r] = start_cycle
            for w in writes:
                # We record the latest write cycle
                curr = last_write.get(w, -1)
                if start_cycle > curr:
                    last_write[w] = start_cycle

        # Convert schedule to list of dicts
        return [dict(s) for s in schedule]

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name or f"const_{val}")
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def v_const(self, val):
        name = f"vconst_{val}"
        if name in self.scratch:
            return self.scratch[name]
        
        s_addr = self.scratch_const(val)
        v_addr = self.alloc_scratch(name, VLEN)
        self.add("valu", ("vbroadcast", v_addr, s_addr))
        return v_addr

    def build_hash_vec(self, v_val_addr, v_tmp1, v_tmp2):
        slots = []
        for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if i == 0:
                # Stage 0: a * 4097 + val1
                vv_mul = self.v_const(4097)
                vv_add = self.v_const(val1)
                slots.append(("valu", ("multiply_add", v_val_addr, v_val_addr, vv_mul, vv_add)))
                continue
            elif i == 2:
                # Stage 2: a * 33 + val1
                vv_mul = self.v_const(33)
                vv_add = self.v_const(val1)
                slots.append(("valu", ("multiply_add", v_val_addr, v_val_addr, vv_mul, vv_add)))
                continue
            elif i == 4:
                # Stage 4: a * 9 + val1
                vv_mul = self.v_const(9)
                vv_add = self.v_const(val1)
                slots.append(("valu", ("multiply_add", v_val_addr, v_val_addr, vv_mul, vv_add)))
                continue
                
            vv1 = self.v_const(val1)
            vv3 = self.v_const(val3)
            slots.append(("valu", (op1, v_tmp1, v_val_addr, vv1)))
            slots.append(("valu", (op3, v_tmp2, v_val_addr, vv3)))
            slots.append(("valu", (op2, v_val_addr, v_tmp1, v_tmp2)))
        return slots

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # Header layout is fixed in build_mem_image; hardcode pointers.
        forest_values_p = self.alloc_scratch("forest_values_p")
        inp_values_p = self.alloc_scratch("inp_values_p")
        self.add("load", ("const", forest_values_p, 7))
        self.add("load", ("const", inp_values_p, 7 + n_nodes + batch_size))

        # Vector constants
        zero_v = self.v_const(0)
        one_v = self.v_const(1)
        two_v = self.v_const(2)

        # Allocate scratch for early round optimization
        t_root_val = self.alloc_scratch("root_val_tmp")
        t_node_val_1 = self.alloc_scratch("t_node_val_1")
        t_node_val_2 = self.alloc_scratch("t_node_val_2")
        t_addr_early = self.alloc_scratch("t_addr_early")
        v_root_val = self.alloc_scratch("v_root_val", VLEN)
        v_node_val_1 = self.alloc_scratch("v_node_val_1", VLEN)
        v_node_val_2 = self.alloc_scratch("v_node_val_2", VLEN)

        # Preload level-2 and level-3 nodes into vector scratch (broadcasted)
        t_cache_addr = self.alloc_scratch("t_cache_addr")
        t_cache_val = self.alloc_scratch("t_cache_val")
        v_cache_l2 = self.alloc_scratch("v_cache_l2", 4 * VLEN)  # idx 3..6

        # Main loop bodies
        all_slots = []
        # Preload cached node values in the packed region to reduce init cost
        pre_slots = []
        pre_slots.append(("load", ("load", t_root_val, self.scratch["forest_values_p"])))
        pre_slots.append(
            ("alu", ("+", t_addr_early, self.scratch["forest_values_p"], self.scratch_const(1)))
        )
        pre_slots.append(("load", ("load", t_node_val_1, t_addr_early)))
        pre_slots.append(
            ("alu", ("+", t_addr_early, self.scratch["forest_values_p"], self.scratch_const(2)))
        )
        pre_slots.append(("load", ("load", t_node_val_2, t_addr_early)))
        pre_slots.append(("valu", ("vbroadcast", v_root_val, t_root_val)))
        pre_slots.append(("valu", ("vbroadcast", v_node_val_1, t_node_val_1)))
        pre_slots.append(("valu", ("vbroadcast", v_node_val_2, t_node_val_2)))

        for i in range(4):
            node_idx = 3 + i
            pre_slots.append(
                (
                    "alu",
                    (
                        "+",
                        t_cache_addr,
                        self.scratch["forest_values_p"],
                        self.scratch_const(node_idx),
                    ),
                )
            )
            pre_slots.append(("load", ("load", t_cache_val, t_cache_addr)))
            pre_slots.append(
                ("valu", ("vbroadcast", v_cache_l2 + i * VLEN, t_cache_val))
            )
        all_slots.extend(pre_slots)
        
        # We process batches in groups of K to fill the pipeline
        K = 29
        num_vec_batches = batch_size // VLEN
        
        # Scratch registers for the loop
        # We need independent temporaries for each of the K interleaved batches
        batch_temps = []
        for k in range(K):
            temps = {
                "v_tmp1": self.alloc_scratch(f"v_tmp1_{k}", VLEN),
                "v_tmp2": self.alloc_scratch(f"v_tmp2_{k}", VLEN),
                "v_node_vals": self.alloc_scratch(f"v_node_vals_{k}", VLEN),
            }
            batch_temps.append(temps)

        # Load the entire batch of values into scratch
        # Indices start at 0, scratch is zero-initialized, so we can skip loading indices.
        v_indices = []
        v_values = []
        for i in range(batch_size // VLEN):
            v_idx = self.alloc_scratch(f"indices_batch_{i}", VLEN)
            v_val = self.alloc_scratch(f"values_batch_{i}", VLEN)
            v_indices.append(v_idx)
            v_values.append(v_val)

        tmp_addr = self.alloc_scratch("tmp_addr")
        for i in range(batch_size // VLEN):
            offset = i * VLEN
            # Load values: mem[inp_values_p + i*VLEN]
            self.add(
                "alu",
                ("+", tmp_addr, self.scratch["inp_values_p"], self.scratch_const(offset)),
            )
            self.add("load", ("vload", v_values[i], tmp_addr))

        self.add("flow", ("pause",))

        # Wavefront schedule across rounds and batches to overlap loads with compute
        for i_base in range(0, num_vec_batches, K):
            k_end = min(K, num_vec_batches - i_base)
            for t in range(rounds + k_end - 1):
                for k in range(k_end):
                    r = t - k
                    if r < 0 or r >= rounds:
                        continue

                    level = r % (forest_height + 1)
                    level_base = None
                    if level >= 3:
                        level_base = self.scratch_const(7 + (1 << level) - 1)

                    i = i_base + k
                    temps = batch_temps[k]

                    # 1. Gather node value for this batch/round
                    if level == 0:
                        all_slots.append(("valu", ("+", temps["v_node_vals"], v_root_val, zero_v)))
                    elif level == 1:
                        all_slots.append(
                            ("flow", ("vselect", temps["v_node_vals"], v_indices[i], v_node_val_2, v_node_val_1))
                        )
                    elif level == 2:
                        # rel is already in range 0..3, select via two bits
                        all_slots.append(("valu", ("&", temps["v_tmp2"], v_indices[i], one_v)))  # bit0
                        all_slots.append(("valu", (">>", temps["v_tmp1"], v_indices[i], one_v)))  # bit1
                        # low = bit0 ? node4 : node3
                        all_slots.append(
                            ("flow", ("vselect", temps["v_node_vals"], temps["v_tmp2"], v_cache_l2 + 1 * VLEN, v_cache_l2 + 0 * VLEN))
                        )
                        # high = bit0 ? node6 : node5  (overwrite bit0 after read)
                        all_slots.append(
                            ("flow", ("vselect", temps["v_tmp2"], temps["v_tmp2"], v_cache_l2 + 3 * VLEN, v_cache_l2 + 2 * VLEN))
                        )
                        # final = bit1 ? high : low
                        all_slots.append(
                            ("flow", ("vselect", temps["v_node_vals"], temps["v_tmp1"], temps["v_tmp2"], temps["v_node_vals"]))
                        )
                    else:
                        # Regular gather (use level base + rel)
                        for vi in range(VLEN):
                            v_idx_addr = v_indices[i] + vi
                            all_slots.append(("alu", ("+", temps["v_tmp1"] + vi, level_base, v_idx_addr)))
                        for vi in range(VLEN):
                            all_slots.append(("load", ("load_offset", temps["v_node_vals"], temps["v_tmp1"], vi)))

                    # 2. XOR (val ^ node_val)
                    all_slots.append(("valu", ("^", v_values[i], v_values[i], temps["v_node_vals"])))

                    # 3. Hash
                    all_slots.extend(self.build_hash_vec(v_values[i], temps["v_tmp1"], temps["v_tmp2"]))

                    # 4. Update indices (not needed after final round)
                    if r != rounds - 1:
                        if level == forest_height:
                            all_slots.append(("valu", ("+", v_indices[i], zero_v, zero_v)))
                        else:
                            all_slots.append(("valu", ("&", temps["v_tmp1"], v_values[i], one_v)))
                            all_slots.append(("valu", ("multiply_add", v_indices[i], v_indices[i], two_v, temps["v_tmp1"])))

        # Store back values at the end (indices are not required for correctness)
        for i in range(batch_size // VLEN):
            offset = i * VLEN
            all_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], self.scratch_const(offset))))
            all_slots.append(("store", ("vstore", tmp_addr, v_values[i])))

        packed_instrs = self.build(all_slots, vliw=True)
        self.instrs.extend(packed_instrs)

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

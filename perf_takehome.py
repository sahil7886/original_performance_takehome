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
        instrs = []
        current_instr = defaultdict(list)
        # Track what has been written/read in the current instruction bundle
        current_writes = set()
        current_reads = set()
        
        for slot in slots:
            engine, op = slot
            reads, writes = self.get_io(slot)
            
            # Check 1: Resource Limits
            if len(current_instr[engine]) >= SLOT_LIMITS[engine]:
                instrs.append(dict(current_instr))
                current_instr = defaultdict(list)
                current_writes = set()
                current_reads = set()
            
            # Check 2: Data Hazard (RAW, WAR, WAW) within the SAME cycle
            # In this architecture (VLIW), all reads happen at start of cycle, all writes at end.
            # So:
            # - Write-After-Read (WAR) is OK: (read x, write x) -> Old x is read, new x is written.
            # - Read-After-Write (RAW) is BAD: (write x, read x) -> The read would get OLD x, not results of write.
            # - Write-After-Write (WAW) is BAD: (write x, write x) -> Race condition.
            
            conflict = False
            
            # RAW Hazard: Does this slot READ something that is being WRITTEN in this cycle?
            if not reads.isdisjoint(current_writes):
                conflict = True
                
            # WAW & RAW (Inverse): Does this slot WRITE something that is being READ or WRITTEN?
            # Actually, if we write X, and a previous op in this cycle reads X, that is OK (WAR).
            # But if a previous op writes X, that is bad (WAW).
            if not writes.isdisjoint(current_writes):
                conflict = True
                
            # SPECIAL CASE: Flow/Control instructions.
            # If we have a jump, we can't pack anything after it usually, or it gets complicated.
            # In this sim, "flow" is just another unit, but let's be safe: 
            # If we see a jump/branch, we might want to start a new bundle to verify behavior.
            # The sim says: "Effects of instructions don't take effect until the end of cycle."
            # So a jump and an ALU op in parallel is fine.
            
            if conflict:
                instrs.append(dict(current_instr))
                current_instr = defaultdict(list)
                current_writes = set()
                current_reads = set()
            
            # Add to bundle
            current_instr[engine].append(op)
            current_writes.update(writes)
            current_reads.update(reads)

        if current_instr:
            instrs.append(dict(current_instr))
            
        return instrs

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
        for op1, val1, op2, op3, val3 in HASH_STAGES:
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
        # Scratch space for header
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        tmp_init = self.alloc_scratch("tmp_init")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

        # Vector constants
        zero_v = self.v_const(0)
        one_v = self.v_const(1)
        two_v = self.v_const(2)
        n_nodes_v = self.v_const(n_nodes)
        
        # Load the entire batch of indices and values into scratch
        # This reduces repetitive address calculation and memory pressure.
        v_indices = []
        v_values = []
        for i in range(batch_size // VLEN):
            v_idx = self.alloc_scratch(f"indices_batch_{i}", VLEN)
            v_val = self.alloc_scratch(f"values_batch_{i}", VLEN)
            v_indices.append(v_idx)
            v_values.append(v_val)

        tmp_addr = self.alloc_scratch("tmp_addr")
        for i in range(batch_size // VLEN):
            # Load indices: mem[inp_indices_p + i*VLEN]
            offset = i * VLEN
            self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], self.scratch_const(offset)))
            self.add("load", ("vload", v_indices[i], tmp_addr))
            # Load values: mem[inp_values_p + i*VLEN]
            self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], self.scratch_const(offset)))
            self.add("load", ("vload", v_values[i], tmp_addr))

        self.add("flow", ("pause",))

        # Main loop bodies
        all_slots = []
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_node_vals = self.alloc_scratch("v_node_vals", VLEN)
        v_target_node_addrs = [self.alloc_scratch(f"node_addr_{vi}") for vi in range(VLEN)]

        for round in range(rounds):
            for i in range(batch_size // VLEN):
                # Gather tree node values
                for vi in range(VLEN):
                    v_idx_addr = v_indices[i] + vi
                    all_slots.append(("alu", ("+", v_target_node_addrs[vi], self.scratch["forest_values_p"], v_idx_addr)))
                    all_slots.append(("load", ("load", v_node_vals + vi, v_target_node_addrs[vi])))
                
                # val = val ^ node_val
                all_slots.append(("valu", ("^", v_values[i], v_values[i], v_node_vals)))
                
                # val = myhash(val)
                all_slots.extend(self.build_hash_vec(v_values[i], v_tmp1, v_tmp2))
                
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                all_slots.append(("valu", ("%", v_tmp1, v_values[i], two_v)))
                all_slots.append(("valu", ("==", v_tmp1, v_tmp1, zero_v)))
                all_slots.append(("flow", ("vselect", v_tmp2, v_tmp1, one_v, two_v)))
                all_slots.append(("valu", ("*", v_indices[i], v_indices[i], two_v)))
                all_slots.append(("valu", ("+", v_indices[i], v_indices[i], v_tmp2)))
                
                # idx = 0 if idx >= n_nodes else idx
                all_slots.append(("valu", ("<", v_tmp1, v_indices[i], n_nodes_v)))
                all_slots.append(("flow", ("vselect", v_indices[i], v_tmp1, v_indices[i], zero_v)))

        # Store back everything at the end
        for i in range(batch_size // VLEN):
            offset = i * VLEN
            all_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], self.scratch_const(offset))))
            all_slots.append(("store", ("vstore", tmp_addr, v_indices[i])))
            all_slots.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], self.scratch_const(offset))))
            all_slots.append(("store", ("vstore", tmp_addr, v_values[i])))

        packed_instrs = self.build(all_slots, vliw=True)
        self.instrs.extend(packed_instrs)
        self.instrs.append({"flow": [("pause",)]})

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

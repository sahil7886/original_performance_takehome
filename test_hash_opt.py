
import unittest
from problem import HASH_STAGES

def myhash_original(a: int) -> int:
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }
    def r(x): return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
    return a

def myhash_optimized(a: int) -> int:
    def r(x): return x % (2**32)
    
    # Stage 0
    # Original: (+, val1, +, <<, 12)
    # a * 4097 + val1
    val1_0 = HASH_STAGES[0][1]
    a = r(a * 4097 + val1_0)

    # Stage 1
    # Original: (^, val1, ^, >>, 19)
    # (a ^ val1) ^ (a >> 19)
    val1_1 = HASH_STAGES[1][1]
    a = r((a ^ val1_1) ^ (a >> 19))

    # Stage 2
    # (+, val1, +, <<, 5)
    # a * 33 + val1
    val1_2 = HASH_STAGES[2][1]
    a = r(a * 33 + val1_2)

    # Stage 3
    # (+, val1, ^, <<, 9)
    # (a + val1) ^ (a << 9)
    val1_3 = HASH_STAGES[3][1]
    a = r((a + val1_3) ^ (a << 9))

    # Stage 4
    # (+, val1, +, <<, 3)
    # a * 9 + val1
    val1_4 = HASH_STAGES[4][1]
    a = r(a * 9 + val1_4)

    # Stage 5
    # (^, val1, ^, >>, 16)
    # (a ^ val1) ^ (a >> 16)
    val1_5 = HASH_STAGES[5][1]
    a = r((a ^ val1_5) ^ (a >> 16))

    return a

class TestHash(unittest.TestCase):
    def test_hash_equivalence(self):
        import random
        for _ in range(100):
            val = random.randint(0, 2**32 - 1)
            h1 = myhash_original(val)
            h2 = myhash_optimized(val)
            self.assertEqual(h1, h2, f"Failed for {val}")

if __name__ == "__main__":
    unittest.main()

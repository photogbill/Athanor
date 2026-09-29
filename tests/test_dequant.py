"""Decoding stored rows — checked against blocks built by hand, so the test
does not depend on the code it checks."""

import struct
import unittest

import numpy as np

from athanor.gguf import GGMLType
from athanor.gguf.dequant import dequantize_row


def f16(x):
    return np.float16(x).tobytes()


class Decoding(unittest.TestCase):

    def test_f32_f16_bf16(self):
        v = np.array([1.5, -2.25, 0.0, 3.0], np.float32)
        self.assertTrue(np.array_equal(dequantize_row(v.tobytes(), GGMLType.F32, 4), v))
        self.assertTrue(np.array_equal(dequantize_row(v.astype(np.float16).tobytes(), GGMLType.F16, 4), v))
        bf = (v.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        self.assertTrue(np.array_equal(dequantize_row(bf, GGMLType.BF16, 4), v))

    def test_q8_0_block(self):
        q = np.arange(-16, 16, dtype=np.int8)
        raw = f16(0.5) + q.tobytes()
        out = dequantize_row(raw, GGMLType.Q8_0, 32)
        self.assertTrue(np.array_equal(out, q.astype(np.float32) * 0.5))

    def test_q4_0_block_low_nibbles_first(self):
        # values 0..15 in low nibbles, 15..0 in high nibbles; stored minus 8
        lo = np.arange(16, dtype=np.uint8)
        hi = np.arange(15, -1, -1, dtype=np.uint8)
        raw = f16(2.0) + (lo | (hi << 4)).astype(np.uint8).tobytes()
        out = dequantize_row(raw, GGMLType.Q4_0, 32)
        want = np.concatenate([lo.astype(np.float32) - 8, hi.astype(np.float32) - 8]) * 2.0
        self.assertTrue(np.array_equal(out, want))

    def test_q4_1_block_has_a_minimum(self):
        lo = np.full(16, 3, np.uint8)
        hi = np.full(16, 1, np.uint8)
        raw = f16(0.25) + f16(-1.0) + (lo | (hi << 4)).tobytes()
        out = dequantize_row(raw, GGMLType.Q4_1, 32)
        self.assertTrue(np.allclose(out[:16], 3 * 0.25 - 1.0))
        self.assertTrue(np.allclose(out[16:], 1 * 0.25 - 1.0))

    def test_other_types_need_gguf_py_and_say_so(self):
        try:
            import gguf  # noqa: F401
            self.skipTest("gguf-py is installed; the fallback is exercised instead")
        except ImportError:
            pass
        from athanor.gguf.dequant import NoDequantizer
        with self.assertRaisesRegex(NoDequantizer, "gguf-py"):
            dequantize_row(b"\0" * 144, GGMLType.Q4_K, 256)


if __name__ == "__main__":
    unittest.main()

"""The GGUF reader and writer — spike S5: a header read and written again is
the same bytes, on every file we can find."""

import os
import struct
import unittest
from pathlib import Path

import numpy as np

from athanor import gguf
from athanor.gguf import Array, GGMLType, GGUFError, GGUFWriter, ValueType
from fixtures import VOCABS, REAL_MODEL, TempDir, all_extra_vocabs, tiny_gguf


class ReadWrite(unittest.TestCase):

    def test_what_is_written_is_what_is_read(self):
        with TempDir() as d:
            p = tiny_gguf(d / "t.gguf", template="{{ messages }}")
            g = gguf.read(p)
            self.assertEqual(g.version, 3)
            self.assertEqual(g.architecture, "llama")
            self.assertEqual(g.get("llama.context_length"), 4096)
            self.assertEqual(g.get("tokenizer.ggml.tokens")[:3], ["<unk>", "<s>", "</s>"])
            self.assertEqual(len(g.get("tokenizer.ggml.token_type")), 64)
            self.assertEqual(g.get("tokenizer.chat_template"), "{{ messages }}")
            t = g.tensor("token_embd.weight")
            self.assertEqual(t.shape, (32, 64))
            self.assertEqual(t.nbytes, 32 * 64 * 4)
            self.assertEqual(t.n_rows, 64)
            self.assertEqual(gguf.validate(g), [])

    def test_header_round_trips_byte_for_byte(self):
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            rt = gguf.round_trip_check(g)
            self.assertTrue(rt["identical"], rt)
            a, b = gguf.copy_stream_hash(g)
            self.assertEqual(a, b)

    def test_the_check_can_fail(self):
        """Negative control: a changed value is noticed."""
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            g.kvs[1].value = "Tinz"
            self.assertFalse(gguf.round_trip_check(g)["identical"])

    def test_rewriting_reproduces_the_file(self):
        with TempDir() as d:
            src = gguf.read(tiny_gguf(d / "a.gguf", alignment=64))
            w = GGUFWriter(version=src.version)
            for kv in src.kvs:
                w.add(kv.key, kv.value, kv.type)        # general.alignment sets the layout
            self.assertEqual(w.alignment, 64)
            for t in src.tensors:
                w.add_tensor(t.name, t.shape, t.ggml_type, src.tensor_bytes(t))
            out = w.write_file(d / "b.gguf")
            self.assertEqual((d / "a.gguf").read_bytes(), out.read_bytes())

    def test_alignment_is_honoured(self):
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf", alignment=64))
            self.assertEqual(g.alignment, 64)
            self.assertEqual(g.data_offset % 64, 0)
            for t in g.tensors:
                self.assertEqual(t.offset % 64, 0)
            self.assertEqual(g.file_size % 64, 0, "the last tensor is padded too")

    def test_strings_that_are_not_utf8_survive(self):
        bad = b"caf\xe9".decode("utf-8", "surrogateescape")
        with TempDir() as d:
            w = GGUFWriter()
            w.add("general.architecture", "llama")
            w.add("x.bad", bad)
            w.add("x.list", [bad, "ok"], ValueType.ARRAY, ValueType.STRING)
            p = w.write_file(d / "t.gguf")
            g = gguf.read(p)
            self.assertEqual(g.get("x.bad").encode("utf-8", "surrogateescape"), b"caf\xe9")
            self.assertTrue(gguf.round_trip_check(g)["identical"])

    def test_every_value_type_round_trips(self):
        with TempDir() as d:
            w = GGUFWriter()
            for vt, val in [(ValueType.UINT8, 200), (ValueType.INT8, -5), (ValueType.UINT16, 60000),
                            (ValueType.INT16, -300), (ValueType.UINT32, 4_000_000_000),
                            (ValueType.INT32, -2), (ValueType.FLOAT32, 0.1),
                            (ValueType.UINT64, 2**63), (ValueType.INT64, -2**40),
                            (ValueType.FLOAT64, 0.1)]:
                w.add(f"x.{vt.name.lower()}", val, vt)
            w.add("x.bool", True)
            w.add("x.nested", Array(ValueType.ARRAY, [Array(ValueType.INT32, np.array([1, 2], "<i4")),
                                                       Array(ValueType.STRING, ["a"])]))
            g = gguf.read(w.write_file(d / "t.gguf"))
            self.assertEqual(g.get("x.uint64"), 2**63)
            self.assertEqual(g.get("x.int16"), -300)
            self.assertAlmostEqual(g.get("x.float32"), 0.1, places=6)
            self.assertEqual(g.get("x.float64"), 0.1)
            self.assertIs(g.get("x.bool"), True)
            self.assertEqual(g.get("x.nested"), [[1, 2], ["a"]])
            self.assertTrue(gguf.round_trip_check(g)["identical"])

    def test_big_endian_files_read_and_round_trip(self):
        with TempDir() as d:
            w = GGUFWriter(endian=">")
            w.add("general.architecture", "llama")
            w.add("x.n", 7, ValueType.UINT32)
            w.add("x.arr", [1.5, 2.5], ValueType.ARRAY, ValueType.FLOAT32)
            g = gguf.read(w.write_file(d / "be.gguf"))
            self.assertEqual(g.endian, ">")
            self.assertEqual(g.get("x.n"), 7)
            self.assertEqual(list(g.get("x.arr")), [1.5, 2.5])
            self.assertTrue(gguf.round_trip_check(g)["identical"])


class RefusesAndReports(unittest.TestCase):

    def test_never_overwrites(self):
        with TempDir() as d:
            p = tiny_gguf(d / "t.gguf")
            before = p.read_bytes()
            with self.assertRaises(FileExistsError):
                tiny_gguf(p)
            self.assertEqual(p.read_bytes(), before)
            self.assertFalse((d / "t.gguf.part").exists())

    def test_not_a_gguf(self):
        with TempDir() as d:
            p = d / "x.gguf"
            p.write_bytes(b"NOPE" + b"\0" * 64)
            with self.assertRaisesRegex(GGUFError, "not a GGUF"):
                gguf.read(p)

    def test_version_1_is_refused_plainly(self):
        with TempDir() as d:
            p = d / "v1.gguf"
            p.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\0" * 32)
            with self.assertRaisesRegex(GGUFError, "version 1"):
                gguf.read(p)

    def test_a_truncated_header(self):
        with TempDir() as d:
            full = tiny_gguf(d / "t.gguf").read_bytes()
            p = d / "cut.gguf"
            p.write_bytes(full[:200])
            with self.assertRaisesRegex(GGUFError, "past the end of the file|truncated"):
                gguf.read(p)

    def test_validate_finds_overlap_and_misalignment(self):
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            g.tensors[1].offset = g.tensors[0].offset + 4
            problems = " ".join(gguf.validate(g))
            self.assertIn("not aligned", problems)
            self.assertIn("overlap", problems)

    def test_an_unknown_tensor_type_is_kept_not_fatal(self):
        with TempDir() as d:
            p = tiny_gguf(d / "t.gguf")
            g = gguf.read(p)
            g.tensors[0].ggml_type = 99
            self.assertEqual(g.tensors[0].type_name, "type99")
            self.assertIsNone(g.tensors[0].nbytes)
            self.assertIn("unknown", " ".join(gguf.validate(g)))


class ReviewFindings(unittest.TestCase):
    """Defects found by the independent review of 0.1, each pinned."""

    def test_the_alignment_key_is_the_layout(self):
        with TempDir() as d:
            w = GGUFWriter()
            w.add("general.architecture", "llama")
            w.add("general.alignment", 64, ValueType.UINT32)
            w.add_tensor("a", (8,), GGMLType.F32, np.zeros(8, np.float32))
            w.add_tensor("b", (8,), GGMLType.F32, np.zeros(8, np.float32))
            g = gguf.read(w.write_file(d / "t.gguf"))
            self.assertEqual(g.alignment, 64)
            self.assertEqual(g.tensors[1].offset, 64)
            self.assertEqual(gguf.validate(g), [])

    def test_a_bad_alignment_is_refused(self):
        for value, vtype in ((48, ValueType.UINT32), (64, ValueType.UINT64), (0, ValueType.UINT32)):
            with self.assertRaises(GGUFError):
                GGUFWriter().add("general.alignment", value, vtype)
        with self.assertRaises(GGUFError):
            GGUFWriter(alignment=48)

    def test_a_file_whose_alignment_is_not_uint32_is_refused_as_llama_cpp_does(self):
        with TempDir() as d:
            w = GGUFWriter()
            w.add("general.architecture", "llama")
            w.kvs.append(gguf.KV("general.alignment", ValueType.UINT64, np.uint64(32)))
            p = d / "t.gguf"
            with open(p, "wb") as f:
                w.write(f)
            with self.assertRaisesRegex(GGUFError, "UINT32"):
                gguf.read(p)

    def test_an_existing_part_file_is_left_alone(self):
        with TempDir() as d:
            part = d / "t.gguf.part"
            part.write_bytes(b"someone else's")
            with self.assertRaises(FileExistsError):
                tiny_gguf(d / "t.gguf")
            self.assertEqual(part.read_bytes(), b"someone else's")
            self.assertFalse((d / "t.gguf").exists())

    def test_a_failed_write_removes_its_part_file(self):
        with TempDir() as d:
            w = GGUFWriter()
            w.add("general.architecture", "llama")

            def boom():
                yield b"\0" * 16
                raise RuntimeError("disk full")

            w.add_tensor("a", (8,), GGMLType.F32, boom)
            with self.assertRaises(RuntimeError):
                w.write_file(d / "t.gguf")
            self.assertEqual(list(d.iterdir()), [])

    def test_a_file_that_appears_mid_write_is_not_overwritten(self):
        with TempDir() as d:
            target = d / "t.gguf"
            w = GGUFWriter()
            w.add("general.architecture", "llama")
            real_write = w.write

            def racing_write(stream):
                target.write_bytes(b"precious")      # someone else wins the race
                return real_write(stream)

            w.write = racing_write
            with self.assertRaises(FileExistsError):
                w.write_file(target)
            self.assertEqual(target.read_bytes(), b"precious")
            self.assertFalse((d / "t.gguf.part").exists())

    def test_damaged_lengths_fail_fast_with_a_reason(self):
        with TempDir() as d:
            full = tiny_gguf(d / "t.gguf").read_bytes()
            bad = bytearray(full)
            # the first key's length (offset 24) claims ~1 GB
            bad[24:32] = struct.pack("<Q", 1 << 30)
            p = d / "bad.gguf"
            p.write_bytes(bytes(bad))
            with self.assertRaisesRegex(GGUFError, "past the end of the file"):
                gguf.read(p)

    def test_tiny_and_malformed_files_are_gguf_errors(self):
        with TempDir() as d:
            short = d / "s.gguf"
            short.write_bytes(b"GGUF\x03\x00")
            with self.assertRaisesRegex(GGUFError, "too short"):
                gguf.read(short)
            w = GGUFWriter()
            w.add("x.arr", [1], ValueType.ARRAY, ValueType.INT32)
            raw = bytearray()
            import io
            buf = io.BytesIO()
            w.write(buf)
            raw = bytearray(buf.getvalue())
            i = raw.index(b"x.arr") + len(b"x.arr") + 4   # past the key and its type
            raw[i:i + 4] = struct.pack("<I", 99)            # an element type GGUF lacks
            p = d / "e.gguf"
            p.write_bytes(bytes(raw))
            with self.assertRaisesRegex(GGUFError, "element type 99"):
                gguf.read(p)

    def test_offsets_must_be_packed_in_order(self):
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            g.tensors[1].offset += 32   # aligned, not overlapping, but not where llama.cpp expects
            self.assertIn("llama.cpp expects", " ".join(gguf.validate(g)))

    def test_a_partial_block_is_named_as_such(self):
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            g.tensors[0].ggml_type = int(GGMLType.Q4_K)   # 32 is not a whole Q4_K block (256)
            self.assertIn("not a whole number of Q4_K blocks", " ".join(gguf.validate(g)))


def _ggml_lib():
    """ggml's own GGUF reader, from the installed binding — the oracle."""
    try:
        import llama_cpp
    except Exception:
        return None
    import ctypes
    import glob
    base = Path(llama_cpp.__file__).parent / "lib"
    names = glob.glob(str(base / "*ggml-base*"))
    if not names:
        return None
    lib = ctypes.CDLL(names[0])

    class Params(ctypes.Structure):
        _fields_ = [("no_alloc", ctypes.c_bool), ("ctx", ctypes.c_void_p)]

    lib.gguf_init_from_file.restype = ctypes.c_void_p
    lib.gguf_init_from_file.argtypes = [ctypes.c_char_p, Params]
    for fn, rt in (("gguf_get_data_offset", ctypes.c_size_t), ("gguf_get_alignment", ctypes.c_size_t),
                   ("gguf_get_n_tensors", ctypes.c_int64)):
        getattr(lib, fn).restype = rt
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    lib.gguf_get_tensor_offset.restype = ctypes.c_size_t
    lib.gguf_get_tensor_offset.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.gguf_free.argtypes = [ctypes.c_void_p]

    def check(path):
        ctx = lib.gguf_init_from_file(os.fsencode(str(path)), Params(True, None))
        if not ctx:
            return None
        out = {"data_offset": lib.gguf_get_data_offset(ctx), "alignment": lib.gguf_get_alignment(ctx),
               "offsets": [lib.gguf_get_tensor_offset(ctx, i)
                           for i in range(lib.gguf_get_n_tensors(ctx))]}
        lib.gguf_free(ctx)
        return out
    return check


class GgmlAgrees(unittest.TestCase):
    """What Athanor writes, ggml's own reader accepts — with the same layout."""

    def setUp(self):
        self.check = _ggml_lib()
        if self.check is None:
            self.skipTest("llama-cpp-python (with its ggml library) is not installed here")

    def test_files_athanor_writes(self):
        with TempDir() as d:
            cases = [tiny_gguf(d / "a.gguf"), tiny_gguf(d / "b.gguf", alignment=64),
                     tiny_gguf(d / "c.gguf", emb_type=GGMLType.Q8_0),
                     tiny_gguf(d / "d.gguf", extra=[("general.alignment", 128, ValueType.UINT32)])]
            for p in cases:
                with self.subTest(p.name):
                    theirs = self.check(p)
                    self.assertIsNotNone(theirs, "ggml refused a file Athanor wrote")
                    ours = gguf.read(p)
                    self.assertEqual(theirs["data_offset"], ours.data_offset)
                    self.assertEqual(theirs["alignment"], ours.alignment)
                    self.assertEqual(theirs["offsets"], [t.offset for t in ours.tensors])

    def test_the_oracle_can_refuse(self):
        """Negative control: a file llama.cpp must refuse, it refuses."""
        with TempDir() as d:
            g = gguf.read(tiny_gguf(d / "t.gguf"))
            g.tensors[1].offset += 32
            head = gguf.serialize_header(g.version, g.endian, g.kvs, g.tensors)
            p = d / "bad.gguf"
            p.write_bytes(head + b"\0" * 4096)
            self.assertIsNone(self.check(p))


class RealFiles(unittest.TestCase):
    """Every real GGUF we can reach round-trips — the S5 evidence."""

    def test_the_bundled_vocabularies(self):
        for name, p in VOCABS.items():
            g = gguf.read(p)
            self.assertTrue(gguf.round_trip_check(g)["identical"], name)
            self.assertGreater(len(g.get("tokenizer.ggml.tokens")), 30000, name)

    def test_llama_cpps_whole_vocabulary_set(self):
        files = all_extra_vocabs()
        if not files:
            self.skipTest("set ATHANOR_TEST_VOCAB_DIR to a llama.cpp models/ folder")
        for p in files:
            with self.subTest(p.name):
                self.assertTrue(gguf.round_trip_check(gguf.read(p))["identical"])

    def test_a_real_model(self):
        if not REAL_MODEL:
            self.skipTest("set ATHANOR_TEST_MODEL to a GGUF model to check it")
        g = gguf.read(REAL_MODEL)
        self.assertEqual(gguf.validate(g), [])
        self.assertTrue(gguf.round_trip_check(g)["identical"])
        if os.environ.get("ATHANOR_TEST_WHOLE_FILE"):
            a, b = gguf.copy_stream_hash(g)
            self.assertEqual(a, b)


class AgainstGgufPy(unittest.TestCase):
    """When llama.cpp's own gguf-py is importable, both sides must agree."""

    def setUp(self):
        try:
            import gguf as G  # noqa: F401
        except Exception:
            self.skipTest("gguf-py not installed (pip install gguf) — optional cross-check")
        self.G = G

    def test_same_bytes_as_gguf_py_writes(self):
        G = self.G
        rng = np.random.default_rng(0)
        with TempDir() as d:
            ref = str(d / "ref.gguf")
            w = G.GGUFWriter(ref, "llama")
            w.add_uint32("llama.context_length", 4096)
            w.add_array("x.arr", [1, 2, 3])
            w.add_string("x.s", "héllo")
            w.add_tensor("a", rng.standard_normal((3, 64)).astype(np.float32))
            w.add_tensor("b", rng.standard_normal((5, 32)).astype(np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            g = gguf.read(ref)
            mine = GGUFWriter(version=g.version)
            for kv in g.kvs:
                mine.add(kv.key, kv.value, kv.type)
            for t in g.tensors:
                mine.add_tensor(t.name, t.shape, t.ggml_type, g.tensor_bytes(t))
            out = mine.write_file(d / "mine.gguf")
            self.assertEqual(Path(ref).read_bytes(), out.read_bytes())
            r = G.GGUFReader(ref)
            for ours, theirs in zip(g.tensors, r.tensors):
                self.assertEqual(ours.abs_offset, int(theirs.data_offset))
                self.assertEqual(ours.nbytes, int(theirs.n_bytes))


if __name__ == "__main__":
    unittest.main()

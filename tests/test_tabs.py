"""The four file tabs: Inspect, Tokenize, Compare, Template."""

import unittest

import numpy as np

from athanor.gguf import GGMLType
from athanor.tabs import compare as C
from athanor.tabs import inspect as I
from athanor.tabs import template as TT
from athanor.tabs import tokenize as K
from athanor.tabs.common import marker_candidates, turn_end
from fixtures import TempDir, VOCABS, extra_vocab, needs_llama, tiny_gguf


def by_check(result, check):
    return [f for f in result["findings"] if f["check"] == check]


class InspectHeaderOnly(unittest.TestCase):
    """Everything that needs no llama.cpp."""

    def test_anatomy_is_labelled(self):
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "Tiny-8k.gguf", ctx=8192)), run_llama=False)
            a = r["anatomy"]
            self.assertEqual(a["architecture"]["label"], "DECLARED")
            self.assertEqual(a["context_length"]["value"], 8192)
            self.assertEqual(a["bits_per_weight"]["label"], "ESTIMATE")
            self.assertEqual(a["parameters"]["value"], 64 * 32 + 32 + 32)
            self.assertEqual(r["tokenizer"]["declared"]["n_tokens"]["value"], 64)
            self.assertIsNone(r["tokenizer"]["measured"])
            self.assertEqual(by_check(r, "llama-cpp")[0]["status"], "skipped")

    def test_no_chat_template_is_a_warning(self):
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "t.gguf")), run_llama=False)
            self.assertEqual(by_check(r, "chat-template")[0]["status"], "warn")

    def test_any_row_count_but_one_per_token_is_a_problem(self):
        """llama.cpp creates token_embd as {n_embd, n_vocab} and refuses any
        other shape (llama-model-loader.cpp, check_tensor_dims) — extra
        'padding' rows included."""
        with TempDir() as d:
            few = I.inspect(str(tiny_gguf(d / "few.gguf", emb_rows=60)), run_llama=False)
            pad = I.inspect(str(tiny_gguf(d / "pad.gguf", emb_rows=96)), run_llama=False)
            ok = I.inspect(str(tiny_gguf(d / "ok.gguf")), run_llama=False)
            self.assertEqual(by_check(few, "vocab-rows")[0]["status"], "problem")
            self.assertEqual(by_check(pad, "vocab-rows")[0]["status"], "problem")
            self.assertEqual(by_check(ok, "vocab-rows")[0]["status"], "ok")

    def test_the_name_against_the_header(self):
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "Tiny-128k.gguf", ctx=4096)), run_llama=False)
            self.assertEqual(by_check(r, "name-vs-header")[0]["status"], "warn")
            ok = I.inspect(str(tiny_gguf(d / "Tiny-4k.gguf", ctx=4096)), run_llama=False)
            self.assertEqual(by_check(ok, "name-vs-header"), [])

    def test_parameter_counts_in_names_are_not_contexts(self):
        self.assertEqual(I._num_in_name("pythia-70m SmolLM-135M Q4_K_M 8B"), [])
        self.assertEqual([n for _t, n in I._num_in_name("Qwen-1M 32K-ctx")], [1048576, 32768])

    def test_a_bpe_without_a_pretokenizer(self):
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "t.gguf", tokenizer_model="gpt2")), run_llama=False)
            self.assertEqual(by_check(r, "pre-tokenizer")[0]["status"], "problem")

    def test_problems_sort_first(self):
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "t.gguf", emb_rows=60)), run_llama=False)
            order = [f["status"] for f in r["findings"]]
            self.assertEqual(order[0], "problem")
            self.assertEqual(order, sorted(order, key=["problem", "warn", "info", "skipped", "ok"].index))

    def test_metadata_lists_every_key(self):
        with TempDir() as d:
            rows = I.metadata(str(tiny_gguf(d / "t.gguf")), max_items=4)
            keys = [r["key"] for r in rows]
            self.assertIn("tokenizer.ggml.tokens", keys)
            toks = next(r for r in rows if r["key"] == "tokenizer.ggml.tokens")
            self.assertEqual((toks["length"], len(toks["preview"])), (64, 4))


@needs_llama
class InspectWithLlama(unittest.TestCase):

    def test_llama_cpps_own_warnings_become_findings(self):
        r = I.inspect(str(VOCABS["gpt-neox"]))
        msgs = " ".join(f["message"] for f in by_check(r, "llama-cpp-log"))
        self.assertIn("missing pre-tokenizer", msgs)
        self.assertNotIn("CONSIDER REGENERATING", msgs, "the banner is decoration")
        self.assertEqual(r["tokenizer"]["measured"]["type"]["value"], "BPE")

    def test_the_models_own_template_is_checked(self):
        r = I.inspect(str(VOCABS["phi-3"]))
        checks = {f["check"]: f["status"] for f in r["findings"]}
        self.assertEqual(checks.get("own-template/turn-end"), "ok")
        self.assertEqual(checks.get("own-template/marker-is-special"), "ok")
        self.assertEqual(checks.get("own-template/bos-once"), "ok",
                         "phi-3's template writes BOS once; both real runners send one")


class TemplateWithoutAModel(unittest.TestCase):

    def test_renders_and_lists_markers(self):
        r = TT.analyse("chatml")
        self.assertIn("<|im_start|>assistant\n", r["render"])
        self.assertIsNone(r["tokens"])
        self.assertEqual([m["marker"] for m in r["markers"]], ["<|im_start|>", "<|im_end|>"])

    def test_turn_end_is_what_follows_a_reply(self):
        self.assertEqual(turn_end(TT.templates.get("chatml").text), "<|im_end|>\n")
        self.assertEqual(turn_end(TT.templates.get("llama3").text), "<|eot_id|>")

    def test_marker_candidates(self):
        t = "{{'[INST] ' + x + ' [/INST]'}}<start_of_turn></s><s><br><|eot_id|>"
        self.assertEqual(marker_candidates(t), ["[INST]", "[/INST]", "<start_of_turn>", "</s>",
                                                "<s>", "<|eot_id|>"])

    def test_a_template_that_will_not_render(self):
        r = TT.analyse("{{ raise_exception('System role not supported') }}")
        self.assertEqual(by_check(r, "renders")[0]["status"], "problem")

    def test_the_library(self):
        lib = TT.library()
        self.assertEqual(len(lib), 55)
        self.assertTrue(all(t["label"] for t in lib))


@needs_llama
class TemplateWithAModel(unittest.TestCase):

    def test_the_right_template_is_clean(self):
        r = TT.analyse("phi3", model=str(VOCABS["phi-3"]))
        statuses = {f["check"]: f["status"] for f in r["findings"]}
        self.assertEqual(statuses["marker-is-special"], "ok")
        self.assertEqual(statuses["turn-end"], "ok")
        kinds = {s["kind"] for s in r["segments"]}
        self.assertTrue({"bos", "control", "text"} <= kinds, kinds)

    def test_a_mismatched_template_is_caught(self):
        r = TT.analyse("chatml", model=str(VOCABS["llama-spm"]))
        marker = by_check(r, "marker-is-special")
        self.assertTrue(marker and all(f["status"] == "problem" for f in marker))
        self.assertEqual(by_check(r, "turn-end")[0]["status"], "problem")

    def test_side_by_side(self):
        r = TT.side_by_side("phi3", "chatml", model=str(VOCABS["phi-3"]))
        self.assertFalse(r["identical"])
        self.assertTrue(r["differences"])
        self.assertIsNotNone(r["token_difference"])


@needs_llama
class Tokenize(unittest.TestCase):

    def test_the_ruler(self):
        text = "The meeting moved to Thursday. Встреча перенесена на четверг."
        r = K.compare_text([str(VOCABS["llama-spm"]), str(VOCABS["gpt-neox"])], text, context=32768)
        self.assertEqual(len(r["models"]), 2)
        for m in r["models"]:
            self.assertEqual(m["n_tokens"]["label"], "MEASURED")
            self.assertEqual(m["ruler"]["words"]["label"], "ESTIMATE")
            self.assertIn("CYRILLIC", m["by_script"])
            self.assertGreater(m["ruler"]["words"]["value"], 1000)

    def test_worst_splits_worst_first(self):
        r = K.worst_splits([str(VOCABS["llama-spm"])])
        worst = [row["tokens_per_char_worst"] for row in r["strings"]]
        self.assertEqual(worst, sorted(worst, reverse=True))
        self.assertEqual(len(r["strings"]), len(K.ANALYST_STRINGS))

    def test_fertility_table(self):
        r = K.fertility_table([str(VOCABS["llama-spm"]), str(VOCABS["phi-3"])], "hello there world")
        self.assertEqual(len(r["rows"]), 2)


class Scripts(unittest.TestCase):

    def test_words_by_script(self):
        w = K.words_by_script("hello мир 世界 سلام 42")
        self.assertEqual(set(w), {"LATIN", "CYRILLIC", "CJK", "ARABIC", "DIGITS"})

    def test_cjk_counts_per_character(self):
        self.assertEqual(K._count_words("会议改到"), 4)
        self.assertEqual(K._count_words("two words"), 2)


class ReviewFindingsTabs(unittest.TestCase):

    def test_words_keep_their_combining_marks(self):
        self.assertEqual(list(K.iter_words("عَبْدُ الرَّحْمٰن")), ["عَبْدُ", "الرَّحْمٰن"])
        self.assertEqual(list(K.iter_words("हिन्दी भाषा")), ["हिन्दी", "भाषा"])
        self.assertEqual(list(K.iter_words("Cafe\u0301 well-known l’homme")),
                         ["Cafe\u0301", "well-known", "l’homme"])
        self.assertEqual(K._count_words("می‌کنم"), 1, "ZWNJ joins a Persian word")
        self.assertEqual(K.words_by_script("हिन्दी भाषा"), {"DEVANAGARI": ["हिन्दी", "भाषा"]})

    def test_an_embedding_model_is_not_held_to_chat_checks(self):
        from athanor.gguf import ValueType
        with TempDir() as d:
            r = I.inspect(str(tiny_gguf(d / "e.gguf", extra=[("llama.pooling_type", 1,
                                                                ValueType.UINT32)])),
                          run_llama=False)
            checks = {f["check"] for f in r["findings"]}
            self.assertIn("embedder", checks)
            self.assertNotIn("chat-template", checks)

    def test_an_explicit_default_pretokenizer_is_not_a_missing_one(self):
        with TempDir() as d:
            missing = I.inspect(str(tiny_gguf(d / "m.gguf", tokenizer_model="gpt2")), run_llama=False)
            explicit = I.inspect(str(tiny_gguf(d / "x.gguf", tokenizer_model="gpt2", pre="default")),
                                 run_llama=False)
            self.assertEqual(by_check(missing, "pre-tokenizer")[0]["status"], "problem")
            self.assertEqual(by_check(explicit, "pre-tokenizer")[0]["status"], "info")

    def test_corpus_truncation_is_reported(self):
        with TempDir() as d:
            (d / "a.txt").write_bytes("\ufeffhello ".encode("utf-8") * 100)
            (d / "b.md").write_text("world", encoding="utf-8")
            c = K.load_corpus(d, max_bytes=50)
            self.assertTrue(c["truncated"])
            self.assertEqual((c["files"], c["files_read"], c["bytes_read"]), (2, 1, 50))
            full = K.load_corpus(d)
            self.assertFalse(full["truncated"])
            self.assertTrue(full["text"].startswith("hello"), "the BOM is dropped")

    def test_lineage_across_a_grown_vocabulary(self):
        from fixtures import tiny_tokens
        rng = np.random.default_rng(1)
        base = rng.standard_normal((64, 32)).astype(np.float32)
        grown = np.vstack([base + 0.01 * rng.standard_normal(base.shape).astype(np.float32),
                           rng.standard_normal((2, 32)).astype(np.float32)])
        with TempDir() as d:
            a = tiny_gguf(d / "a.gguf", emb=base)
            b = tiny_gguf(d / "b.gguf", emb=grown, tokens=tiny_tokens(64) + ["<|x|>", "<|y|>"])
            lin = C.compare(str(a), str(b))["lineage"]
            self.assertIn("same base", lin["verdict"])
            self.assertEqual(lin["vocabulary_rows"], {"a": 64, "b": 66})


@needs_llama
class ReviewFindingsTemplates(unittest.TestCase):

    def statuses(self, r):
        return {(f["check"], f["status"]) for f in r["findings"]}

    def test_llama2_on_llama2s_own_tokenizer_is_not_a_problem(self):
        r = TT.analyse("llama2-sys", model=str(VOCABS["llama-spm"]))
        self.assertNotIn("problem", {s for _c, s in self.statuses(r)}, r["findings"])
        self.assertIn(("turn-end", "ok"), self.statuses(r), "the space before </s> is formatting")

    def test_a_control_token_format_on_the_wrong_tokenizer_is(self):
        r = TT.analyse("mistral-v7", model=str(VOCABS["llama-spm"]))
        self.assertIn(("marker-is-special", "problem"), self.statuses(r))

    def test_a_template_that_refuses_a_system_message(self):
        t = ("{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not "
             "supported') }}{% endif %}{% for m in messages %}{{ m['content'] }}\n{% endfor %}")
        r = TT.analyse(t, model=str(VOCABS["llama-spm"]))
        self.assertIn(("renders", "info"), self.statuses(r))
        self.assertIsNotNone(r["render"])
        self.assertTrue(all(m["role"] != "system" for m in r["messages"]))

    def test_bos_counted_as_each_runner_builds_the_prompt(self):
        r = TT.analyse("llama3", model=str(VOCABS["phi-3"]))
        b = r["bos"]
        self.assertEqual((b["expected"], b["llama_cpp_python"], b["llama_server"]), (1, 1, 1))
        self.assertEqual(b["naive_runner"], 2)
        self.assertIn(("bos-once", "ok"), self.statuses(r))
        self.assertIn(("bos-naive", "info"), self.statuses(r))


class Compare(unittest.TestCase):

    def _pair(self, d, noise=None, other=False, q8=False):
        rng = np.random.default_rng(3)
        base = rng.standard_normal((64, 32)).astype(np.float32)
        a = tiny_gguf(d / "a.gguf", emb=base)
        if other:
            emb = rng.standard_normal((64, 32)).astype(np.float32)
        else:
            emb = base + (noise or 0) * rng.standard_normal(base.shape).astype(np.float32)
        b = tiny_gguf(d / "b.gguf", emb=emb, emb_type=GGMLType.Q8_0 if q8 else GGMLType.F32)
        return str(a), str(b)

    def test_identical(self):
        with TempDir() as d:
            a, _ = self._pair(d)
            r = C.compare(a, a)
            self.assertTrue(r["same_header"])
            self.assertEqual(r["lineage"]["verdict"], "identical rows")

    def test_a_fine_tune(self):
        with TempDir() as d:
            r = C.compare(*self._pair(d, noise=0.02))
            self.assertIn("same base", r["lineage"]["verdict"])
            self.assertEqual(r["lineage"]["verdict_label"], "ESTIMATE")

    def test_a_requantization(self):
        with TempDir() as d:
            r = C.compare(*self._pair(d, q8=True))
            self.assertIn("same base", r["lineage"]["verdict"])
            self.assertEqual(r["tensors"]["n_type_differs"], 1)

    def test_unrelated_weights(self):
        with TempDir() as d:
            r = C.compare(*self._pair(d, other=True))
            self.assertEqual(r["lineage"]["verdict"], "not the same weights")

    def test_tokenizer_and_metadata_differences(self):
        r = C.compare(str(VOCABS["llama-spm"]), str(VOCABS["phi-3"]))
        t = r["tokenizer"]
        self.assertFalse(t["identical"])
        self.assertEqual(t["ids_that_differ"], 0, "phi-3 extends Llama's list")
        self.assertEqual(t["only_in_b"]["count"], 64)
        self.assertEqual(t["special_ids_differ"]["eos_token_id"], {"a": 2, "b": 32000})
        self.assertIn("general.architecture", [c["key"] for c in r["metadata"]["changed"]])

    def test_overlap_matrix(self):
        paths = [str(VOCABS["llama-spm"]), str(VOCABS["phi-3"]), str(VOCABS["gpt-neox"])]
        r = C.overlap_matrix(paths)
        self.assertTrue(r["identical_vocabulary"][0][0])
        self.assertFalse(r["identical_vocabulary"][0][1])
        self.assertGreater(r["overlap"][0][1], 0.99)
        self.assertLess(r["overlap"][0][2], 0.5)


if __name__ == "__main__":
    unittest.main()

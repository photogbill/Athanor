"""The template library — every one of the 55 held to llama.cpp's own C++
output, byte for byte (tests/data/llama_cpp_chat_reference.json, made by
compiling llama-chat.cpp at b11093 and calling it)."""

import json
import unittest

from athanor import templates as T
from athanor.gguf import ValueType
from fixtures import DATA, TempDir, tiny_gguf

T_STRING = ValueType.STRING

REFERENCE = json.loads((DATA / "llama_cpp_chat_reference.json").read_text(encoding="utf-8"))


def render(text, messages, gen=True, bos=""):
    return T.render(text, messages, add_generation_prompt=gen, bos_token=bos, eos_token="</s>")


class EveryTemplateIsLlamaCpps(unittest.TestCase):

    def test_the_reference_is_the_pinned_tag(self):
        self.assertEqual(REFERENCE["llama_cpp_tag"], T.LLAMA_CPP_TAG)
        self.assertEqual(sorted(REFERENCE["expected"]), sorted(T.names()))
        self.assertEqual(len(T.names()), 55)

    def test_byte_for_byte_against_llama_cpp(self):
        wrong = []
        checked = 0
        for t in T.all_templates():
            for conv in REFERENCE["conversations"]:
                for gen in (True, False):
                    want = REFERENCE["expected"][t.name][conv["id"]]["gen" if gen else "nogen"]
                    checked += 1
                    if render(t.text, conv["messages"], gen) != want:
                        wrong.append(f"{t.name} / {conv['id']} / gen={gen}")
        self.assertEqual(wrong, [], "templates that differ from llama.cpp")
        self.assertGreaterEqual(checked, 55 * 10 * 2, "the comparison must actually run")

    def test_the_comparison_catches_a_one_character_drift(self):
        text = T.get("llama3").text.replace("<|eot_id|>", "<|eot_id|> ")
        conv = REFERENCE["conversations"][0]
        self.assertNotEqual(render(text, conv["messages"]),
                            REFERENCE["expected"]["llama3"][conv["id"]]["gen"])

    def test_bos_is_one_token_in_front_and_nothing_else(self):
        conv = REFERENCE["conversations"][2]["messages"]
        for t in T.all_templates():
            self.assertEqual(render(t.text, conv, bos="<BOS>"), "<BOS>" + render(t.text, conv), t.name)

    def test_image_parts_render_as_their_text(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "What is on the plate?"},
                                              {"type": "text", "text": "<__media__>"}]}]
        text = [{"role": "user", "content": "What is on the plate?<__media__>"}]
        for t in T.all_templates():
            self.assertEqual(render(t.text, parts), render(t.text, text), t.name)


class Detection(unittest.TestCase):

    def test_detection_agrees_with_llama_cpp(self):
        wrong = [(d["id"], d["expected"], T.detect(d["text"])) for d in REFERENCE["detection"]
                 if T.detect(d["text"]) != d["expected"]]
        self.assertEqual(wrong, [])
        self.assertGreaterEqual(len({d["expected"] for d in REFERENCE["detection"]}), 35)


class Environment(unittest.TestCase):
    """The same Jinja the binding builds."""

    def test_tojson_keeps_non_ascii(self):
        out = T.render("{{ messages[0].content | tojson }}", [{"role": "user", "content": "café"}])
        self.assertEqual(out, '"café"')

    def test_generation_tags_pass_through(self):
        out = T.render("{% generation %}X{% endgeneration %}", [])
        self.assertEqual(out, "X")

    def test_loop_controls_and_raise_exception(self):
        out = T.render("{% for m in messages %}{% if loop.index > 1 %}{% break %}{% endif %}"
                       "{{ m.content }}{% endfor %}",
                       [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        self.assertEqual(out, "a")
        with self.assertRaisesRegex(ValueError, "no system"):
            T.render("{{ raise_exception('no system') }}", [])

    def test_the_sandbox_holds(self):
        with self.assertRaises(Exception):
            T.render("{{ messages.__class__.__mro__ }}{{ ''.__class__.__subclasses__() }}", [])


class Resolve(unittest.TestCase):

    def test_names_with_and_without_prefix(self):
        self.assertEqual(T.resolve("chatml")["name"], "llama.cpp/chatml")
        self.assertEqual(T.resolve("llama.cpp/chatml")["text"], T.get("chatml").text)

    def test_the_models_own(self):
        with TempDir() as d:
            from athanor import gguf
            g = gguf.read(tiny_gguf(d / "t.gguf", template="{{ 'hi' }}",
                                    extra=[("tokenizer.chat_template.tool_use", "{{ 'tools' }}",
                                            T_STRING)]))
            self.assertEqual(T.resolve("gguf", gguf=g)["text"], "{{ 'hi' }}")
            self.assertEqual(T.resolve("gguf:tool_use", gguf=g)["text"], "{{ 'tools' }}")
            with self.assertRaisesRegex(ValueError, "has no chat template 'x'"):
                T.resolve("gguf:x", gguf=g)

    def test_a_file_and_text(self):
        with TempDir() as d:
            f = d / "mine.jinja"
            f.write_text("{{ 'x' }}", encoding="utf-8")
            self.assertEqual(T.resolve(str(f))["source"], "file")
        self.assertEqual(T.resolve("{{ 'y' }}")["source"], "text")
        with self.assertRaisesRegex(ValueError, "no template called"):
            T.resolve("no-such-format")


if __name__ == "__main__":
    unittest.main()

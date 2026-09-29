"""Every prompt template llama.cpp ships, as Jinja — the Template tab's library.

llama.cpp carries a reference formatter for every chat format it knows
(``src/llama-chat.cpp``, ``llm_chat_apply_template``): 54 by name, one more
(``dots1``) by detection only. This module is those 55, written as Jinja so
any of them can be rendered, compared, and installed into any model,
whatever its GGUF recommends — a fine-tune's own template is sometimes wrong.

WHY THE JINJA CAN BE TRUSTED
----------------------------
Not written from memory or a model card. Each template is a transcription of
the C++ at llama.cpp ``b11093``, and ``tests/test_templates.py`` holds every
one to the C++'s own output, BYTE FOR BYTE, over a battery of conversations
(system or not, one turn or many, generation prompt on and off, leading and
trailing whitespace, roles the format does not know). The reference strings
were produced by compiling that file and calling it; they are kept in
``tests/data/llama_cpp_chat_reference.json`` with the tag they came from.

Origin: written 2026-09-26 for the Analyst Toolkit (ATK) and released with
Athanor under the MIT licence by its author.

CONVENTIONS
-----------
* Every template starts with ``{{- bos_token -}}``. llama.cpp's own
  formatter leaves BOS to the tokeniser; ``render`` passes an empty
  ``bos_token`` unless asked. The Template tab passes the model's BOS text,
  as llama-cpp-python and llama.cpp's server both do, and then counts the
  BOS each of them would actually send. (ATK passes an empty ``bos_token``
  for a model whose tokenizer adds none; a runner that does not can send a
  BOS such a model never expects.)
* Where the C++ trims a message it uses C's ``isspace`` — space, tab, the
  four line/page breaks. Python's ``strip`` also removes Unicode spaces, so
  the trims here name their characters (``trim(ws)``) rather than differ on
  a non-breaking space.
* A message's content may be a list of parts (a vision path turns each
  image into a text marker); ``text()`` joins the text parts in order.
"""

from __future__ import annotations

from dataclasses import dataclass

#: the llama.cpp tag these were transcribed from and checked against
LLAMA_CPP_TAG = "b11093"

#: Shared by every template: the content macro, C's whitespace for trims,
#: and the BOS every template opens with.
_PRE = (r"""{%- macro text(c) -%}{%- if c is string -%}{{- c -}}"""
        r"""{%- elif c is iterable -%}{%- for p in c -%}"""
        r"""{%- if p is mapping and p['type'] == 'text' -%}{{- p['text'] -}}"""
        r"""{%- endif -%}{%- endfor -%}{%- endif -%}{%- endmacro -%}"""
        r"""{%- set ws = ' \t\n\x0b\x0c\r' -%}"""
        r"""{{- bos_token -}}""")


def _t(body: str) -> str:
    return _PRE + body


@dataclass(frozen=True)
class Template:
    #: llama.cpp's own name for it
    name: str
    #: what the operator sees in the dropdown
    label: str
    #: the model families it is for, as llama.cpp's comments name them
    note: str
    text: str


def _mistral_inst(leading: str, trailing: str, trim_assistant: bool) -> str:
    """mistral-v1 / v3 / v3-tekken: one [INST] per turn, system text folded
    into it, the reply after [/INST] closed by </s>."""
    reply = ("text(m['content'])|trim(ws)" if trim_assistant
             else "text(m['content'])")
    return _t(
        r"""{%- set ns = namespace(inside=false) -%}"""
        r"""{%- for m in messages -%}"""
        r"""{%- if not ns.inside -%}{{- '""" + leading + r"""[INST]"""
        + trailing + r"""' -}}{%- set ns.inside = true -%}{%- endif -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- text(m['content']) + '"""
        + leading + r"""[/INST]' -}}"""
        r"""{%- else -%}{{- '""" + trailing + r"""' + """ + reply
        + r""" + '</s>' -}}{%- set ns.inside = false -%}{%- endif -%}"""
        r"""{%- endfor -%}""")


def _mistral_v7(space: str) -> str:
    return _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '[SYSTEM_PROMPT]""" + space
        + r"""' + text(m['content']) + '[/SYSTEM_PROMPT]' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '[INST]""" + space
        + r"""' + text(m['content']) + '[/INST]' -}}"""
        r"""{%- else -%}{{- '""" + space
        + r"""' + text(m['content']) + '</s>' -}}{%- endif -%}"""
        r"""{%- endfor -%}""")


def _llama2(system: bool, bos_inside: bool, strip: bool) -> str:
    content = "text(m['content'])|trim(ws)" if strip else "text(m['content'])"
    sys_part = (r"""'<<SYS>>\n' + c + '\n<</SYS>>\n\n'""" if system
                else r"""c + '\n'""")
    reopen = "<s>[INST] " if bos_inside else "[INST] "
    return _t(
        r"""{%- set ns = namespace(inside=true) -%}{{- '[INST] ' -}}"""
        r"""{%- for m in messages -%}{%- set c = """ + content + r""" -%}"""
        r"""{%- if not ns.inside -%}{%- set ns.inside = true -%}{{- '"""
        + reopen + r"""' -}}{%- endif -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- """ + sys_part + r""" -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- c + ' [/INST]' -}}"""
        r"""{%- else -%}{{- c + '</s>' -}}{%- set ns.inside = false -%}"""
        r"""{%- endif -%}{%- endfor -%}""")


def _exaone(tool: bool) -> str:
    tool_part = (r"""{%- elif m['role'] == 'tool' -%}{{- '[|tool|]' + c + """
                 r"""'[|endofturn|]\n' -}}""") if tool else ""
    return _t(
        r"""{%- for m in messages -%}{%- set c = text(m['content'])|trim(ws) -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '[|system|]' + c + '[|endofturn|]\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '[|user|]' + c + '\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- '[|assistant|]' + c + '[|endofturn|]\n' -}}"""
        + tool_part +
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '[|assistant|]' -}}{%- endif -%}""")


def _granite(v4: bool) -> str:
    if v4:
        head = (r"""{%- if m['role'] == 'assistant_tool_call' -%}"""
                r"""{{- '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' -}}"""
                r"""{%- else -%}{{- '<|start_of_role|>' + m['role'] + '<|end_of_role|>' -}}"""
                r"""{%- endif -%}""")
    else:
        head = (r"""{{- '<|start_of_role|>' + m['role'] + '<|end_of_role|>' -}}"""
                r"""{%- if m['role'] == 'assistant_tool_call' -%}"""
                r"""{{- '<|tool_call|>' -}}{%- endif -%}""")
    return _t(
        r"""{%- for m in messages -%}""" + head +
        r"""{{- text(m['content']) + '<|end_of_text|>\n' -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}"""
        r"""{{- '<|start_of_role|>assistant<|end_of_role|>' -}}{%- endif -%}""")


def _bailing(think: bool) -> str:
    return _t(
        r"""{%- for m in messages -%}"""
        r"""{%- set r = 'HUMAN' if m['role'] == 'user' else m['role']|upper -%}"""
        r"""{{- '<role>' + r + '</role>' + text(m['content']) -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<role>ASSISTANT</role>"""
        + ("<think>" if think else "") + r"""' -}}{%- endif -%}""")


def _vicuna(orca: bool) -> str:
    system = (r"""'SYSTEM: ' + text(m['content']) + '\n'""" if orca
              else r"""text(m['content']) + '\n\n'""")
    return _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- """ + system + r""" -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'USER: ' + text(m['content']) + '\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- 'ASSISTANT: ' + text(m['content']) + '</s>\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'ASSISTANT:' -}}{%- endif -%}""")


def _simple(open_: str, between: str, close: str, gen: str,
            trim: bool = False) -> str:
    """Every message the same shape: open + role + between + content +
    close; then the generation prompt."""
    content = "text(m['content'])|trim(ws)" if trim else "text(m['content'])"
    return _t(
        r"""{%- for m in messages -%}{{- '""" + open_ + r"""' + m['role'] + '"""
        + between + r"""' + """ + content + r""" + '""" + close
        + r"""' -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '""" + gen
        + r"""' -}}{%- endif -%}""")


_TEMPLATES: tuple[Template, ...] = (
    Template("chatml", "ChatML", "Qwen 1.5–3, Hermes, Yi, OpenHermes and "
             "most fine-tunes",
             _simple(r"<|im_start|>", r"\n", r"<|im_end|>\n",
                     r"<|im_start|>assistant\n")),
    Template("llama2", "Llama 2 (system folded in)",
             "Llama-2-chat without <<SYS>>", _llama2(False, False, False)),
    Template("llama2-sys", "Llama 2 with <<SYS>>", "Llama-2-chat",
             _llama2(True, False, False)),
    Template("llama2-sys-bos", "Llama 2 with <<SYS>>, BOS between turns",
             "Llama-2 variants that repeat <s>", _llama2(True, True, False)),
    Template("llama2-sys-strip", "Llama 2 with <<SYS>>, trimmed",
             "Llama-2 variants that strip messages",
             _llama2(True, False, True)),
    Template("mistral-v1", "Mistral v1", "Mistral 7B v0.1/v0.2, Mixtral 8x7B",
             _mistral_inst(" ", " ", False)),
    Template("mistral-v3", "Mistral v3", "Mistral 7B v0.3, Mixtral 8x22B",
             _mistral_inst("", " ", True)),
    Template("mistral-v3-tekken", "Mistral v3-Tekken",
             "Mistral NeMo, Ministral", _mistral_inst("", "", False)),
    Template("mistral-v7", "Mistral v7", "Mistral Large 2411, Pixtral Large",
             _mistral_v7(" ")),
    Template("mistral-v7-tekken", "Mistral v7-Tekken",
             "Mistral Small 3.x, Magistral, Devstral (llama.cpp's plain "
             "form — no default system prompt)", _mistral_v7("")),
    Template("phi3", "Phi-3", "Phi-3 / Phi-3.5",
             _simple(r"<|", r"|>\n", r"<|end|>\n", r"<|assistant|>\n")),
    Template("phi4", "Phi-4", "Phi-4",
             _simple(r"<|im_start|>", r"<|im_sep|>", r"<|im_end|>",
                     r"<|im_start|>assistant<|im_sep|>")),
    Template("falcon3", "Falcon 3", "Falcon 3",
             _simple(r"<|", r"|>\n", r"\n", r"<|assistant|>\n")),
    Template("zephyr", "Zephyr", "Zephyr, StableLM 2",
             _simple(r"<|", r"|>\n", r"<|endoftext|>\n", r"<|assistant|>\n")),
    Template("monarch", "Monarch", "AlphaMonarch-7B", _t(
        r"""{%- for m in messages -%}"""
        r"""{{- ('' if loop.first else '<s>') + m['role'] + '\n' + text(m['content']) + '</s>\n' -}}"""
        r"""{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<s>assistant\n' -}}{%- endif -%}""")),
    Template("gemma", "Gemma", "Gemma 1, 2 and 3 (system folded into the "
             "first user turn)", _t(
        r"""{%- set ns = namespace(sys='') -%}{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{%- set ns.sys = ns.sys + text(m['content'])|trim(ws) -%}"""
        r"""{%- else -%}{%- set r = 'model' if m['role'] == 'assistant' else m['role'] -%}"""
        r"""{{- '<start_of_turn>' + r + '\n' -}}"""
        r"""{%- if ns.sys and r != 'model' -%}{{- ns.sys + '\n\n' -}}{%- set ns.sys = '' -%}{%- endif -%}"""
        r"""{{- text(m['content'])|trim(ws) + '<end_of_turn>\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<start_of_turn>model\n' -}}{%- endif -%}""")),
    Template("orion", "Orion", "Orion-14B-Chat", _t(
        r"""{%- set ns = namespace(sys='') -%}{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{%- set ns.sys = ns.sys + text(m['content']) -%}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'Human: ' -}}"""
        r"""{%- if ns.sys -%}{{- ns.sys + '\n\n' -}}{%- set ns.sys = '' -%}{%- endif -%}"""
        r"""{{- text(m['content']) + '\n\nAssistant: </s>' -}}"""
        r"""{%- else -%}{{- text(m['content']) + '</s>' -}}{%- endif -%}{%- endfor -%}""")),
    Template("openchat", "OpenChat", "OpenChat 3.5", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) + '<|end_of_turn|>' -}}"""
        r"""{%- else -%}{{- 'GPT4 Correct ' + m['role'][:1]|upper + m['role'][1:] + ': ' + text(m['content']) + '<|end_of_turn|>' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'GPT4 Correct Assistant:' -}}{%- endif -%}""")),
    Template("vicuna", "Vicuna", "Vicuna 1.1", _vicuna(False)),
    Template("vicuna-orca", "Vicuna (Orca)", "Orca-Vicuna", _vicuna(True)),
    Template("deepseek", "DeepSeek Coder", "deepseek-coder instruct", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '### Instruction:\n' + text(m['content']) + '\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- '### Response:\n' + text(m['content']) + '\n<|EOT|>\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '### Response:\n' -}}{%- endif -%}""")),
    Template("deepseek2", "DeepSeek V2", "DeepSeek-V2", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'User: ' + text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- 'Assistant: ' + text(m['content']) + '<｜end▁of▁sentence｜>' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'Assistant:' -}}{%- endif -%}""")),
    Template("deepseek3", "DeepSeek V3 / R1", "DeepSeek-V3, DeepSeek-R1 and "
             "its distills", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '<｜User｜>' + text(m['content']) -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- '<｜Assistant｜>' + text(m['content']) + '<｜end▁of▁sentence｜>' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<｜Assistant｜>' -}}{%- endif -%}""")),
    Template("deepseek-ocr", "DeepSeek OCR (no template)", "DeepSeek-OCR — "
             "messages are sent as they are", _t(
        r"""{%- for m in messages -%}{{- text(m['content']) -}}{%- endfor -%}""")),
    Template("command-r", "Command-R", "Cohere Command-R / R+ / A", _t(
        r"""{%- for m in messages -%}{%- set c = text(m['content'])|trim(ws) -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + c + '<|END_OF_TURN_TOKEN|>' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + c + '<|END_OF_TURN_TOKEN|>' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' + c + '<|END_OF_TURN_TOKEN|>' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' -}}{%- endif -%}""")),
    Template("llama3", "Llama 3", "Llama 3, 3.1, 3.2, 3.3",
             _simple(r"<|start_header_id|>", r"<|end_header_id|>\n\n",
                     r"<|eot_id|>",
                     r"<|start_header_id|>assistant<|end_header_id|>\n\n",
                     trim=True)),
    Template("chatglm3", "ChatGLM3", "chatglm3-6b", _t(
        r"""{{- '[gMASK]sop' -}}{%- for m in messages -%}"""
        r"""{{- '<|' + m['role'] + '|>\n ' + text(m['content']) -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}""")),
    Template("chatglm4", "GLM-4", "GLM-4, GLM-4-0414", _t(
        r"""{{- '[gMASK]<sop>' -}}{%- for m in messages -%}"""
        r"""{{- '<|' + m['role'] + '|>\n' + text(m['content']) -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|assistant|>\n' -}}{%- endif -%}""")),
    Template("glmedge", "GLM-Edge", "GLM-Edge", _t(
        r"""{%- for m in messages -%}"""
        r"""{{- '<|' + m['role'] + '|>\n' + text(m['content']) -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}""")),
    Template("minicpm", "MiniCPM", "MiniCPM (OpenHermes)", _t(
        r"""{%- for m in messages -%}{%- set c = text(m['content'])|trim(ws) -%}"""
        r"""{%- if m['role'] == 'user' -%}{{- '<用户>' + c + '<AI>' -}}"""
        r"""{%- else -%}{{- c -}}{%- endif -%}{%- endfor -%}""")),
    Template("exaone3", "EXAONE 3", "EXAONE 3.0 / 3.5", _exaone(False)),
    Template("exaone4", "EXAONE 4", "EXAONE 4.0", _exaone(True)),
    Template("exaone-moe", "EXAONE MoE", "EXAONE MoE", _t(
        r"""{%- for m in messages -%}{%- set c = text(m['content'])|trim(ws) -%}"""
        r"""{%- if m['role'] in ['system', 'user', 'assistant', 'tool'] -%}"""
        r"""{{- '<|' + m['role'] + '|>\n' + c + '<|endofturn|>\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|assistant|>\n' -}}{%- endif -%}""")),
    Template("rwkv-world", "RWKV World", "RWKV World (needs \\n\\n as "
             "end-of-turn)", _t(
        r"""{%- for m in messages -%}{%- set c = text(m['content'])|trim(ws) -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- 'System: ' + c + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'User: ' + c + '\n\n' -}}"""
        r"""{%- if loop.last -%}{{- 'Assistant:' -}}{%- endif -%}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- 'Assistant: ' + c + '\n\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}""")),
    Template("granite", "Granite 3.x", "IBM Granite 3.x", _granite(False)),
    Template("granite-4.0", "Granite 4.0", "IBM Granite 4.0", _granite(True)),
    Template("granite-4.1", "Granite 4.1", "IBM Granite 4.1", _granite(True)),
    Template("gigachat", "GigaChat", "GigaChat", _t(
        r"""{%- set has_system = messages|length > 0 and messages[0]['role'] == 'system' -%}"""
        r"""{%- if has_system -%}{{- '<s>' + text(messages[0]['content']) + '<|message_sep|>' -}}"""
        r"""{%- else -%}{{- '<s>' -}}{%- endif -%}"""
        r"""{%- for m in messages -%}{%- if not (loop.first and has_system) -%}"""
        r"""{%- if m['role'] == 'user' -%}{{- 'user<|role_sep|>' + text(m['content']) + '<|message_sep|>available functions<|role_sep|>[]<|message_sep|>' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- 'assistant<|role_sep|>' + text(m['content']) + '<|message_sep|>' -}}"""
        r"""{%- endif -%}{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'assistant<|role_sep|>' -}}{%- endif -%}""")),
    Template("megrez", "Megrez", "Megrez",
             _simple(r"<|role_start|>", r"<|role_end|>", r"<|turn_end|>",
                     r"<|role_start|>assistant<|role_end|>")),
    Template("yandex", "YandexGPT", "YandexGPT (\\n\\n is end-of-turn)", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'user' -%}{{- ' Пользователь: ' + text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- ' Ассистент: ' + text(m['content']) + '\n\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- ' Ассистент:[SEP]' -}}{%- endif -%}""")),
    Template("bailing", "Bailing (Ling)", "Ling / Ring", _bailing(False)),
    Template("bailing-think", "Bailing (Ling), thinking",
             "Ring (opens <think>)", _bailing(True)),
    Template("bailing2", "Bailing 2 (Ling 2.0)", "Ling 2.0", _t(
        r"""{%- if not (messages|length > 0 and messages[0]['role'] == 'system') -%}"""
        r"""{{- '<role>SYSTEM</role>detailed thinking off<|role_end|>' -}}{%- endif -%}"""
        r"""{%- for m in messages -%}"""
        r"""{%- set r = 'HUMAN' if m['role'] == 'user' else m['role']|upper -%}"""
        r"""{{- '<role>' + r + '</role>' + text(m['content']) + '<|role_end|>' -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<role>ASSISTANT</role>' -}}{%- endif -%}""")),
    Template("llama4", "Llama 4", "Llama 4 Scout / Maverick",
             _simple(r"<|header_start|>", r"<|header_end|>\n\n", r"<|eot|>",
                     r"<|header_start|>assistant<|header_end|>\n\n",
                     trim=True)),
    Template("smolvlm", "SmolVLM", "SmolVLM (its BOS is <|im_start|>; not "
             "ChatML)", _t(
        r"""{{- '<|im_start|>' -}}{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- text(m['content']) + '\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'User: ' + text(m['content']) + '<end_of_utterance>\n' -}}"""
        r"""{%- else -%}{{- 'Assistant: ' + text(m['content']) + '<end_of_utterance>\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'Assistant:' -}}{%- endif -%}""")),
    Template("dots1", "dots.llm1", "dots.llm1.inst (llama.cpp finds it by "
             "detection only)", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '<|system|>' + text(m['content']) + '<|endofsystem|>' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '<|userprompt|>' + text(m['content']) + '<|endofuserprompt|>' -}}"""
        r"""{%- else -%}{{- '<|response|>' + text(m['content']) + '<|endofresponse|>' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|response|>' -}}{%- endif -%}""")),
    Template("hunyuan-moe", "Hunyuan MoE", "Hunyuan-A13B", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '<|startoftext|>' + text(m['content']) + '<|extra_4|>' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- text(m['content']) + '<|eos|>' -}}"""
        r"""{%- else -%}{{- '<|startoftext|>' + text(m['content']) + '<|extra_0|>' -}}"""
        r"""{%- endif -%}{%- endfor -%}""")),
    Template("gpt-oss", "gpt-oss (Harmony)", "OpenAI gpt-oss 20B / 120B", _t(
        r"""{%- for m in messages -%}"""
        r"""{{- '<|start|>' + m['role'] + '<|message|>' + text(m['content']) + ('<|return|>' if m['role'] == 'assistant' else '<|end|>') -}}"""
        r"""{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}""")),
    Template("hunyuan-dense", "Hunyuan Dense", "Hunyuan 0.5B–7B", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if loop.first and m['role'] == 'system' -%}{{- text(m['content']) + '<｜hy_place▁holder▁no▁3｜>' -}}{%- endif -%}"""
        r"""{%- if m['role'] == 'assistant' -%}{{- '<｜hy_Assistant｜>' + text(m['content']) + '<｜hy_place▁holder▁no▁2｜>' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '<｜hy_User｜>' + text(m['content']) + '<｜hy_Assistant｜>' -}}"""
        r"""{%- endif -%}{%- endfor -%}""")),
    Template("hunyuan-vl", "Hunyuan VL / OCR", "HunyuanOCR, HunyuanVL", _t(
        r"""{{- '<｜hy_begin▁of▁sentence｜>' -}}{%- for m in messages -%}"""
        r"""{%- if loop.first and m['role'] == 'system' -%}{{- text(m['content']) + '<｜hy_place▁holder▁no▁3｜>' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- text(m['content']) + '<｜hy_User｜>' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- text(m['content']) + '<｜hy_Assistant｜>' -}}"""
        r"""{%- endif -%}{%- endfor -%}""")),
    Template("kimi-k2", "Kimi K2", "Moonshot Kimi-K2", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- '<|im_system|>system<|im_middle|>' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- '<|im_user|>user<|im_middle|>' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- '<|im_assistant|>assistant<|im_middle|>' -}}"""
        r"""{%- elif m['role'] == 'tool' -%}{{- '<|im_system|>tool<|im_middle|>' -}}"""
        r"""{%- endif -%}{{- text(m['content']) + '<|im_end|>' -}}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<|im_assistant|>assistant<|im_middle|>' -}}{%- endif -%}""")),
    Template("seed_oss", "Seed-OSS", "ByteDance Seed-OSS", _t(
        r"""{%- for m in messages -%}"""
        r"""{{- '<seed:bos>' + m['role'] + '\n' + (text(m['content'])|trim(ws) if m['role'] == 'assistant' else text(m['content'])) + '<seed:eos>' -}}"""
        r"""{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '<seed:bos>assistant\n' -}}{%- endif -%}""")),
    Template("grok-2", "Grok 2", "xAI Grok-2", _t(
        r"""{%- for m in messages -%}"""
        r"""{%- if m['role'] == 'system' -%}{{- 'System: ' + text(m['content'])|trim(ws) + '<|separator|>\n\n' -}}"""
        r"""{%- elif m['role'] == 'user' -%}{{- 'Human: ' + text(m['content'])|trim(ws) + '<|separator|>\n\n' -}}"""
        r"""{%- elif m['role'] == 'assistant' -%}{{- 'Assistant: ' + text(m['content']) + '<|separator|>\n\n' -}}"""
        r"""{%- endif -%}{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- 'Assistant:' -}}{%- endif -%}""")),
    Template("pangu-embedded", "openPangu Embedded", "openPangu-Embedded", _t(
        r"""{%- set names = {'system': '系统', 'user': '用户', 'assistant': '助手', 'tool': '工具', 'function': '方法'} -%}"""
        r"""{%- for m in messages -%}"""
        r"""{%- if loop.first and m['role'] != 'system' -%}{{- '[unused9]系统：[unused10]' -}}{%- endif -%}"""
        r"""{%- if m['role'] in names -%}{{- '[unused9]' + names[m['role']] + '：' + text(m['content']) + '[unused10]' -}}{%- endif -%}"""
        r"""{%- endfor -%}"""
        r"""{%- if add_generation_prompt -%}{{- '[unused9]助手：' -}}{%- endif -%}""")),
    Template("solar-open", "Solar Open", "Upstage Solar Open",
             _simple(r"<|begin|>", r"<|content|>", r"<|end|>",
                     r"<|begin|>assistant")),
)

_BY_NAME = {t.name: t for t in _TEMPLATES}


def names() -> list[str]:
    return [t.name for t in _TEMPLATES]


def get(name: str) -> Template | None:
    return _BY_NAME.get(str(name or "").strip())


def all_templates() -> tuple[Template, ...]:
    return _TEMPLATES


# ---------------------------------------------------------------------------
# what a GGUF's own template looks like — llama.cpp's detection, in Python
# ---------------------------------------------------------------------------

def detect(template: str | None) -> str:
    """The llama.cpp name for the format a Jinja template produces, or "".

    A port of ``llm_chat_detect_template`` (llama-chat.cpp, the same tag):
    the same markers, in the same order, so what Athanor says a GGUF's template
    "looks like" is what llama.cpp itself would decide. It reads markers,
    not meaning — a fine-tune's template can carry one family's markers and
    another's layout, which is exactly when the dropdown is for.
    """
    t = str(template or "")
    if not t:
        return ""
    if t in _BY_NAME:
        return t
    has = t.__contains__
    if has("<|im_start|>"):
        if has("<|im_sep|>"):
            return "phi4"
        return "smolvlm" if has("<end_of_utterance>") else "chatml"
    if t.startswith("mistral") or has("[INST]"):
        if has("[SYSTEM_PROMPT]"):
            return "mistral-v7"
        if has("' [INST] ' + system_message") or has("[AVAILABLE_TOOLS]"):
            if has(" [INST]"):
                return "mistral-v1"
            if has('"[INST]"'):
                return "mistral-v3-tekken"
            return "mistral-v3"
        if has("content.strip()"):
            return "llama2-sys-strip"
        if has("bos_token + '[INST]"):
            return "llama2-sys-bos"
        if has("<<SYS>>"):
            return "llama2-sys"
        return "llama2"
    if has("<|assistant|>") and has("<|end|>"):
        return "phi3"
    if has("[gMASK]<sop>"):
        return "chatglm4"
    if has("<|assistant|>") and has("<|user|>"):
        if has("<|tool_declare|>"):
            return "exaone-moe"
        return "falcon3" if has("</s>") else "glmedge"
    if has("<|{{ item['role'] }}|>") and has("<|begin_of_image|>"):
        return "glmedge"
    if has("<|user|>") and has("<|endoftext|>"):
        return "zephyr"
    if has("bos_token + message['role']"):
        return "monarch"
    if has("<start_of_turn>"):
        return "gemma"
    if has("'\\n\\nAssistant: ' + eos_token"):
        return "orion"
    if has("GPT4 Correct "):
        return "openchat"
    if has("USER: ") and has("ASSISTANT: "):
        return "vicuna-orca" if has("SYSTEM: ") else "vicuna"
    if has("### Instruction:") and has("<|EOT|>"):
        return "deepseek"
    if has("<|START_OF_TURN_TOKEN|>") and has("<|USER_TOKEN|>"):
        return "command-r"
    if has("<|start_header_id|>") and has("<|end_header_id|>"):
        return "llama3"
    if has("[gMASK]sop"):
        return "chatglm3"
    if has("<用户>"):
        return "minicpm"
    if has("'Assistant: ' + message['content'] + eos_token"):
        return "deepseek2"
    if has("<｜Assistant｜>") and has("<｜User｜>") and has("<｜end▁of▁sentence｜>"):
        return "deepseek3"
    if has("[|system|]") and has("[|assistant|]") and has("[|endofturn|]"):
        return "exaone4" if has("[|tool|]") else "exaone3"
    if has("rwkv-world") or has(
            "{{- 'User: ' + message['content']|trim + '\\n\\n' -}}"):
        return "rwkv-world"
    if has("<|start_of_role|>"):
        if has("<tool_call>") or has("<tools>"):
            return ("granite-4.0" if has("g4_default_system_message")
                    else "granite-4.1")
        return "granite"
    if has("message['role'] + additional_special_tokens[0] + "
           "message['content'] + additional_special_tokens[1]"):
        return "gigachat"
    if has("<|role_start|>"):
        return "megrez"
    if has(" Ассистент:"):
        return "yandex"
    if has("<role>ASSISTANT</role>") and has("'HUMAN'"):
        return "bailing"
    if has("<role>ASSISTANT</role>") and has('"HUMAN"') and has("<think>"):
        return "bailing-think"
    if (has("<role>ASSISTANT</role>") and has("<role>HUMAN</role>")
            and has("<|role_end|>")):
        return "bailing2"
    if has("<|header_start|>") and has("<|header_end|>"):
        return "llama4"
    if has("<|endofuserprompt|>"):
        return "dots1"
    if has("<|extra_0|>") and has("<|extra_4|>"):
        return "hunyuan-moe"
    if has("<|start|>") and has("<|channel|>"):
        return "gpt-oss"
    if has("<｜hy_Assistant｜>") and has("<｜hy_begin▁of▁sentence｜>"):
        return "hunyuan-vl"
    if has("<｜hy_Assistant｜>") and has("<｜hy_place▁holder▁no▁3｜>"):
        return "hunyuan-dense"
    if has("<|im_assistant|>assistant<|im_middle|>"):
        return "kimi-k2"
    if has("<seed:bos>"):
        return "seed_oss"
    if has("'Assistant: '  + message['content'] + '<|separator|>"):
        return "grok-2"
    if has("[unused9]系统：[unused10]"):
        return "pangu-embedded"
    if has("<|begin|>") and has("<|end|>") and has("<|content|>"):
        return "solar-open"
    return ""

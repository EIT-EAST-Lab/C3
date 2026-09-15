"""Thinking stays off on hybrid-thinking chat templates.

The Qwen3-8B probe of 2026-09-15 (E1, two agents, n=4, 20 questions) produced 400 <think> tags
in 20 buckets: the base Qwen3 template opens every reply with a think block unless it is told
`enable_thinking=False`. The analysis plan measures the second substrate in non-thinking mode,
so the prompt composer passes the switch whenever the template has it and nothing otherwise.
"""

from __future__ import annotations

from c3.mas.rollout_generator import _compose_full_prompt_chat, chat_template_kwargs

QWEN3_HYBRID_TEMPLATE = "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
QWEN_INSTRUCT_TEMPLATE = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"


class _Tokenizer:
    def __init__(self, template):
        self.chat_template = template
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(dict(kwargs))
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if kwargs.get("enable_thinking") is False:
            body += "\nassistant: <think>\n\n</think>\n\n"
        return body


def test_hybrid_template_gets_thinking_off(monkeypatch):
    monkeypatch.delenv("C3_ENABLE_THINKING", raising=False)
    tok = _Tokenizer(QWEN3_HYBRID_TEMPLATE)
    assert chat_template_kwargs(tok) == {"enable_thinking": False}
    out = _compose_full_prompt_chat(tokenizer=tok, system_prompt="be brief", question="1+1", context="")
    assert tok.calls == [{"enable_thinking": False}]
    assert out.endswith("<think>\n\n</think>\n\n")


def test_instruct_template_gets_no_extra_argument(monkeypatch):
    monkeypatch.delenv("C3_ENABLE_THINKING", raising=False)
    tok = _Tokenizer(QWEN_INSTRUCT_TEMPLATE)
    assert chat_template_kwargs(tok) == {}
    _compose_full_prompt_chat(tokenizer=tok, system_prompt="be brief", question="1+1", context="ctx")
    assert tok.calls == [{}]


def test_env_switch_turns_thinking_back_on(monkeypatch):
    monkeypatch.setenv("C3_ENABLE_THINKING", "1")
    assert chat_template_kwargs(_Tokenizer(QWEN3_HYBRID_TEMPLATE)) == {"enable_thinking": True}
    assert chat_template_kwargs(_Tokenizer(QWEN_INSTRUCT_TEMPLATE)) == {}


def test_tokenizer_without_template_attribute_is_left_alone():
    class Bare:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "rendered"

    assert chat_template_kwargs(Bare()) == {}
    assert chat_template_kwargs(None) == {}
    assert _compose_full_prompt_chat(tokenizer=Bare(), system_prompt="s", question="q", context="") == "rendered"

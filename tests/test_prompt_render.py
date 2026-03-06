from __future__ import annotations

from c3.mas.prompt_render import build_render_context, render_role_prompt


def test_prompt_render_inserts_question_context_and_role_outputs() -> None:
    ctx = build_render_context(
        question='What is 2 + 2?',
        role_outputs={'reasoner': 'Add the two integers.'},
        topo_so_far=['reasoner'],
    )
    prompt = 'Q: {question}\nC: {context}\nR: {reasoner_output}\nM: {missing_key}'
    rendered = render_role_prompt(prompt, ctx=ctx)
    assert 'What is 2 + 2?' in rendered
    assert 'Add the two integers.' in rendered
    assert '{missing_key}' not in rendered


def test_prompt_render_returns_raw_prompt_on_unmatched_braces() -> None:
    prompt = 'Malformed { prompt'
    rendered = render_role_prompt(prompt, ctx={'question': 'x'})
    assert rendered == prompt

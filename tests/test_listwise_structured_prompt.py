from __future__ import annotations

import json

from src.prompts import candidate_block, get_prompt_template, history_block


def test_listwise_ranking_v1_structured_ids_renders_json_allowed_list() -> None:
    tid = "id_a"
    allowed = [tid, "id_b"]
    texts = {tid: "Title: A", "id_b": "Title: B"}
    tmpl = get_prompt_template("listwise_ranking_v1_structured_ids")
    j = json.dumps(allowed, ensure_ascii=False)
    s = tmpl.render(
        history_block=history_block([], texts),
        candidate_block=candidate_block(allowed, texts),
        allowed_item_ids=", ".join(allowed),
        allowed_item_ids_json=j,
        topk=2,
    )
    assert j in s
    assert "id_a" in s

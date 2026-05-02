from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    task: str
    template: str

    def render(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "pointwise_yesno_v1": PromptTemplate(
        prompt_id="pointwise_yesno_v1",
        task="pointwise",
        template=(
            "You are evaluating recommendation relevance.\n"
            "User history:\n{history_block}\n\n"
            "Candidate item:\nitem_id={candidate_item_id}\n{candidate_text}\n\n"
            "Return strict JSON with keys recommend (yes/no), confidence (0-1), reason."
        ),
    ),
    "pairwise_preference_v1": PromptTemplate(
        prompt_id="pairwise_preference_v1",
        task="pairwise",
        template=(
            "User history:\n{history_block}\n\n"
            "Item A: item_id={item_a_id}\n{item_a_text}\n\n"
            "Item B: item_id={item_b_id}\n{item_b_text}\n\n"
            "Return strict JSON: preferred_item, confidence, reason."
        ),
    ),
    "listwise_ranking_v1": PromptTemplate(
        prompt_id="listwise_ranking_v1",
        task="listwise",
        template=(
            "Rank candidate items for the user.\n"
            "User history:\n{history_block}\n\n"
            "Candidates:\n{candidate_block}\n\n"
            "Allowed item IDs: {allowed_item_ids}\n"
            "Return strict JSON with ranked_item_ids, topk_item_ids, confidence, reason."
        ),
    ),
    "generative_next_item_v1": PromptTemplate(
        prompt_id="generative_next_item_v1",
        task="generative",
        template=(
            "Given this user history, predict the next item from the candidate set.\n"
            "History:\n{history_block}\n\nCandidates:\n{candidate_block}\n"
            "Return strict JSON with predicted_item_id, confidence, reason."
        ),
    ),
    "topk_confidence_v1": PromptTemplate(
        prompt_id="topk_confidence_v1",
        task="topk_confidence",
        template=(
            "Rank the top {topk} items and state confidence in the full ranking.\n"
            "History:\n{history_block}\n\nCandidates:\n{candidate_block}\n"
            "Return JSON: topk_item_ids, confidence, item_confidences."
        ),
    ),
    "confidence_elicitation_v1": PromptTemplate(
        prompt_id="confidence_elicitation_v1",
        task="confidence_elicitation",
        template=(
            "Assess whether the proposed recommendation is likely correct.\n"
            "History:\n{history_block}\nRecommendation: item_id={candidate_item_id}\n{candidate_text}\n"
            "Return JSON: confidence, uncertainty_reason."
        ),
    ),
    "self_consistency_v1": PromptTemplate(
        prompt_id="self_consistency_v1",
        task="self_consistency",
        template=(
            "Independently solve this recommendation task. Do not copy prior answers.\n"
            "History:\n{history_block}\nCandidates:\n{candidate_block}\n"
            "Return JSON: ranked_item_ids, confidence, reason."
        ),
    ),
    "perturbation_consistency_v1": PromptTemplate(
        prompt_id="perturbation_consistency_v1",
        task="perturbation_consistency",
        template=(
            "The user history has been order-preserving perturbed for robustness analysis.\n"
            "History:\n{history_block}\nCandidates:\n{candidate_block}\n"
            "Return JSON: ranked_item_ids, confidence, reason."
        ),
    ),
    "uncertainty_reranking_v1": PromptTemplate(
        prompt_id="uncertainty_reranking_v1",
        task="uncertainty_reranking",
        template=(
            "Rerank candidates using relevance and uncertainty. Prefer relevant items, but avoid high-risk confident errors.\n"
            "History:\n{history_block}\nCandidates with uncertainty:\n{candidate_block}\n"
            "Return JSON: ranked_item_ids, confidence, reason."
        ),
    ),
    "explanation_secondary_v1": PromptTemplate(
        prompt_id="explanation_secondary_v1",
        task="explanation",
        template=(
            "Explain a recommendation decision for secondary qualitative analysis only.\n"
            "History:\n{history_block}\nCandidate: item_id={candidate_item_id}\n{candidate_text}\n"
            "Return JSON: explanation, confidence."
        ),
    ),
}


def get_prompt_template(prompt_id: str) -> PromptTemplate:
    if prompt_id not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown prompt template id: {prompt_id}")
    return PROMPT_TEMPLATES[prompt_id]


def history_block(history_item_ids: list[str], item_text_lookup: dict[str, str]) -> str:
    if not history_item_ids:
        return "[empty history]"
    return "\n".join(
        f"{idx}. item_id={item_id} {item_text_lookup.get(item_id, '')}".strip()
        for idx, item_id in enumerate(history_item_ids, start=1)
    )


def candidate_block(candidate_item_ids: list[str], item_text_lookup: dict[str, str]) -> str:
    return "\n".join(
        f"{idx}. item_id={item_id} {item_text_lookup.get(item_id, '')}".strip()
        for idx, item_id in enumerate(candidate_item_ids, start=1)
    )

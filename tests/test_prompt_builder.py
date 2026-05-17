from src.llm.prompt_builder import PromptBuilder


def test_history_items_accept_string_entries() -> None:
    sample = {"history_items": ["Cleanser", "Lip balm"]}

    assert PromptBuilder.build_history_block(sample) == "1. Cleanser\n2. Lip balm"


def test_history_items_keep_dict_entries() -> None:
    sample = {
        "history_items": [
            {"title": "Cleanser", "meta": "skin care"},
            {"text": "Fallback text"},
        ]
    }

    assert PromptBuilder.build_history_block(sample) == "1. Cleanser | skin care\n2. Fallback text"

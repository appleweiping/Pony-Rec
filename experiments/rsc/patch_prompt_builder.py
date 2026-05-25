"""Patch prompt_builder.py to add compact ranking block method."""
import sys

path = "src/llm/prompt_builder.py"
with open(path, "r") as f:
    content = f.read()

new_method = '''
    @staticmethod
    def build_candidate_ranking_block_compact(sample: "dict[str, Any]", max_text_chars: int = 0) -> str:
        item_ids = sample.get("candidate_item_ids", [])
        titles = sample.get("candidate_titles", [])
        texts = sample.get("candidate_texts", [])
        if not isinstance(item_ids, list) or len(item_ids) == 0:
            raise ValueError("Candidate ranking sample is missing candidate_item_ids.")
        lines: list = []
        for idx, item_id in enumerate(item_ids, start=1):
            title = str(titles[idx - 1]).strip() if idx - 1 < len(titles) else ""
            if max_text_chars > 0 and idx - 1 < len(texts):
                text_snippet = str(texts[idx - 1]).strip()[:max_text_chars]
                lines.append(f"{idx}. item_id={item_id} | {title} | {text_snippet}")
            else:
                lines.append(f"{idx}. item_id={item_id} | {title}")
        return "\\n".join(lines)

'''

marker = "    def build_pointwise_prompt("
if marker in content and "build_candidate_ranking_block_compact" not in content:
    content = content.replace(marker, new_method + "    def build_pointwise_prompt(")
    with open(path, "w") as f:
        f.write(content)
    print("Patched successfully")
elif "build_candidate_ranking_block_compact" in content:
    print("Already patched")
else:
    print("ERROR: marker not found")
    sys.exit(1)

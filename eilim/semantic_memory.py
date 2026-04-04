from typing import Any


def build_semantic_context(chat_history: list[dict[str, Any]], max_turns: int = 8) -> dict[str, Any]:
    trimmed = chat_history[-max_turns:]

    user_turns = [str(item.get("text", "")).strip() for item in trimmed if item.get("role") == "user"]
    assistant_turns = [
        str(item.get("text", "")).strip() for item in trimmed if item.get("role") == "assistant"
    ]

    user_intents = [_first_sentence(text, 120) for text in user_turns if text]
    assistant_key_points = [_first_sentence(text, 140) for text in assistant_turns if text]

    latest_user_goal = user_intents[-1] if user_intents else ""
    summary_parts: list[str] = []

    if user_intents:
        summary_parts.append("Recent user intents: " + " | ".join(user_intents[-3:]))
    if assistant_key_points:
        summary_parts.append("Recent tutor points: " + " | ".join(assistant_key_points[-3:]))
    if latest_user_goal:
        summary_parts.append(f"Current goal: {latest_user_goal}")

    return {
        "summary": " || ".join(summary_parts),
        "user_intents": user_intents[-5:],
        "assistant_key_points": assistant_key_points[-5:],
        "latest_user_goal": latest_user_goal,
        "turn_count": len(trimmed),
    }


def _first_sentence(text: str, max_len: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    cut_chars = [".", "?", "!", "\n"]
    stops = [cleaned.find(ch) for ch in cut_chars if cleaned.find(ch) != -1]
    end = min(stops) + 1 if stops else len(cleaned)
    sentence = cleaned[:end].strip()

    if len(sentence) > max_len:
        sentence = sentence[: max_len - 3].rstrip() + "..."

    return sentence
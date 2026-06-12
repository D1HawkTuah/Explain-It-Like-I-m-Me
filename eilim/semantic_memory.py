import re
from collections import Counter
from typing import Any


def retrieve_related_turns(chat_history: list[dict[str, Any]], query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the most relevant past turns using lightweight token overlap scoring.

    This provides a simple, dependency-free semantic-memory prototype that can be
    replaced later by real embeddings without changing callers.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for item in chat_history:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        tokens = _tokenize(text)
        if not tokens:
            continue
        score = _cosine_similarity(query_tokens, tokens)
        if score > 0:
            scored.append({"text": text, "role": item.get("role", "user"), "score": score, "source": item.get("source", "")})

    scored.sort(key=lambda entry: entry["score"], reverse=True)
    return scored[:limit]


def _tokenize(text: str) -> Counter[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(words)


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum((left & right).values())
    denominator = (sum(left.values()) * sum(right.values())) ** 0.5
    return numerator / denominator if denominator else 0.0


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

    related_turns = retrieve_related_turns(trimmed, latest_user_goal or " ".join(user_intents[-3:]), limit=3)

    return {
        "summary": " || ".join(summary_parts),
        "user_intents": user_intents[-5:],
        "assistant_key_points": assistant_key_points[-5:],
        "latest_user_goal": latest_user_goal,
        "turn_count": len(trimmed),
        "related_turns": related_turns,
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
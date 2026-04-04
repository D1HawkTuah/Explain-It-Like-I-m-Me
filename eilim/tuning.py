from typing import List, Tuple

from .models import Feedback, UserProfile

_LEVELS = ["beginner", "intermediate", "advanced"]
_MIN_FEEDBACK_FOR_LEVEL_SHIFT = 3
_STYLE_KEYWORDS = {
    "visual": ["visual", "diagram", "chart", "map"],
    "story": ["story", "narrative", "real-life", "example"],
    "code": ["code", "python", "snippet", "pseudocode"],
    "step-by-step": ["step", "slow", "clear", "walkthrough"],
}


def tune_profile_from_feedback(
    profile: UserProfile,
    latest_feedback: Feedback,
    recent_feedback: List[Feedback],
) -> Tuple[UserProfile, List[str]]:
    updates: List[str] = []

    desired_style = _style_from_comment(latest_feedback.comment)
    if desired_style and desired_style != profile.learning_style:
        profile.learning_style = desired_style
        updates.append(f"learning_style -> {desired_style}")

    level = profile.knowledge_level if profile.knowledge_level in _LEVELS else "beginner"
    profile.knowledge_level = level

    history = recent_feedback[-5:] if recent_feedback else [latest_feedback]
    avg_rating = sum(item.rating for item in history) / len(history)

    if len(history) < _MIN_FEEDBACK_FOR_LEVEL_SHIFT:
        updates.append("knowledge_level unchanged (need at least 3 feedback entries)")
        return profile, updates

    if latest_feedback.rating <= 2 or avg_rating <= 2.5:
        lowered = _move_level(profile.knowledge_level, direction=-1)
        if lowered != profile.knowledge_level:
            profile.knowledge_level = lowered
            updates.append(f"knowledge_level -> {lowered} (simplified after low ratings)")
    elif latest_feedback.rating >= 5 or avg_rating >= 4.5:
        raised = _move_level(profile.knowledge_level, direction=1)
        if raised != profile.knowledge_level:
            profile.knowledge_level = raised
            updates.append(f"knowledge_level -> {raised} (increased after strong ratings)")

    return profile, updates


def _move_level(level: str, direction: int) -> str:
    idx = _LEVELS.index(level)
    new_idx = max(0, min(len(_LEVELS) - 1, idx + direction))
    return _LEVELS[new_idx]


def _style_from_comment(comment: str) -> str:
    text = (comment or "").lower()
    if not text:
        return ""

    for style, keywords in _STYLE_KEYWORDS.items():
        if any(word in text for word in keywords):
            return style
    return ""

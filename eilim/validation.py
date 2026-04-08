"""Centralized input validation for user profiles and form inputs."""

from typing import Optional

# Valid choices for profile fields
VALID_KNOWLEDGE_LEVELS = {"beginner", "intermediate", "advanced"}
VALID_LEARNING_STYLES = {"step-by-step", "visual", "story", "code"}
VALID_SURVEY_PREFERENCES = {
    "examples-first",
    "analogy-first",
    "step-by-step",
    "visual-map",
}

# Request size limits (bytes)
MAX_USER_ID_LENGTH = 128
MAX_TOPIC_LENGTH = 2000
MAX_COMMENT_LENGTH = 5000
MAX_DISPLAY_NAME_LENGTH = 256
MAX_INTERESTS_LENGTH = 500
MAX_SELF_EXPLAINER_LENGTH = 2000


def normalize_user_id(user_id: str) -> str:
    """Normalize user_id: strip whitespace, coerce empty to 'guest'."""
    normalized = (user_id or "").strip()[:MAX_USER_ID_LENGTH]
    return normalized if normalized else "guest"


def validate_knowledge_level(level: Optional[str], default: str = "beginner") -> str:
    """Validate knowledge level and return valid choice or default."""
    if not level:
        return default
    normalized = level.lower().strip()
    return normalized if normalized in VALID_KNOWLEDGE_LEVELS else default


def validate_learning_style(style: Optional[str], default: str = "step-by-step") -> str:
    """Validate learning style and return valid choice or default."""
    if not style:
        return default
    normalized = style.lower().strip()
    return normalized if normalized in VALID_LEARNING_STYLES else default


def validate_survey_preference(
    survey: Optional[str], default: str = "examples-first"
) -> str:
    """Validate survey preference and return valid choice or default."""
    if not survey:
        return default
    normalized = survey.lower().strip()
    return normalized if normalized in VALID_SURVEY_PREFERENCES else default


def validate_quiz_score(score_raw: Optional[str], default: int = -1) -> int:
    """Parse quiz score string, bound to [0-3] or return default."""
    if not score_raw or not score_raw.strip():
        return default
    try:
        parsed = int(score_raw.strip())
    except ValueError:
        return default
    return max(0, min(3, parsed))


def validate_rating(rating_raw: Optional[str], default: int = -1) -> int:
    """Parse rating string, bound to [1-5] or return default."""
    if not rating_raw or not rating_raw.strip():
        return default
    try:
        parsed = int(rating_raw.strip())
    except ValueError:
        return default
    return max(1, min(5, parsed)) if parsed >= 1 else default


def parse_csv_field(value: Optional[str], max_length: int = MAX_INTERESTS_LENGTH) -> list[str]:
    """Parse comma-separated field, strip each item, filter empty, enforce length limit."""
    if not value:
        return []
    # Truncate input if too large
    value = value[:max_length]
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_display_name(name: Optional[str], fallback: str = "user", max_length: int = MAX_DISPLAY_NAME_LENGTH) -> str:
    """Validate display name: strip whitespace or use fallback, enforce length limit."""
    if not name:
        return fallback
    cleaned = name.strip()[:max_length]
    return cleaned if cleaned else fallback


def validate_topic(topic: Optional[str], max_length: int = MAX_TOPIC_LENGTH) -> str:
    """Validate topic: strip whitespace, enforce length limit."""
    if not topic:
        return ""
    return topic.strip()[:max_length]


def validate_comment(comment: Optional[str], max_length: int = MAX_COMMENT_LENGTH) -> str:
    """Validate feedback comment: strip whitespace, enforce length limit."""
    if not comment:
        return ""
    return comment.strip()[:max_length]


def validate_self_explainer(sample: Optional[str], max_length: int = MAX_SELF_EXPLAINER_LENGTH) -> str:
    """Validate self-explainer sample: strip whitespace, enforce length limit."""
    if not sample:
        return ""
    return sample.strip()[:max_length]


class InputValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_request_payload_size(data: dict, max_fields: int = 20) -> None:
    """Validate request payload is not suspiciously large."""
    if len(data) > max_fields:
        raise InputValidationError(f"Request has too many fields ({len(data)} > {max_fields})")

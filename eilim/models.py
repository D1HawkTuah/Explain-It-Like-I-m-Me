from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class UserProfile:
    user_id: str
    display_name: str
    knowledge_level: str
    learning_style: str
    interests: List[str] = field(default_factory=list)
    domains_of_focus: List[str] = field(default_factory=list)
    self_explainer_sample: str = ""
    onboarding_survey: str = ""
    calibration_quiz_score: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "UserProfile":
        return UserProfile(
            user_id=payload["user_id"],
            display_name=payload.get("display_name", payload["user_id"]),
            knowledge_level=payload.get("knowledge_level", "beginner"),
            learning_style=payload.get("learning_style", "step-by-step"),
            interests=payload.get("interests", []),
            domains_of_focus=payload.get("domains_of_focus", []),
            self_explainer_sample=payload.get("self_explainer_sample", ""),
            onboarding_survey=payload.get("onboarding_survey", ""),
            calibration_quiz_score=int(payload.get("calibration_quiz_score", -1)),
        )


@dataclass
class Interaction:
    user_id: str
    topic: str
    explanation: str
    domain: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Feedback:
    user_id: str
    topic: str
    rating: int
    comment: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

from eilim.models import Feedback, UserProfile
from eilim.tuning import tune_profile_from_feedback


def test_low_rating_can_lower_level_and_switch_style_from_comment():
    profile = UserProfile(
        user_id="u1",
        display_name="A",
        knowledge_level="advanced",
        learning_style="code",
        interests=[],
        domains_of_focus=[],
    )

    latest = Feedback(user_id="u1", topic="taxes", rating=1, comment="too hard, please make it visual")
    updated, updates = tune_profile_from_feedback(profile, latest, [latest])

    assert updated.knowledge_level == "advanced"
    assert updated.learning_style == "visual"
    assert any("need at least 3 feedback entries" in item for item in updates)


def test_high_ratings_can_raise_level():
    profile = UserProfile(
        user_id="u1",
        display_name="A",
        knowledge_level="beginner",
        learning_style="step-by-step",
        interests=[],
        domains_of_focus=[],
    )

    history = [
        Feedback(user_id="u1", topic="math", rating=5, comment="great"),
        Feedback(user_id="u1", topic="math", rating=5, comment="great"),
        Feedback(user_id="u1", topic="math", rating=4, comment="good depth"),
    ]
    latest = history[-1]

    updated, _ = tune_profile_from_feedback(profile, latest, history)
    assert updated.knowledge_level == "intermediate"


def test_low_ratings_shift_level_when_history_threshold_met():
    profile = UserProfile(
        user_id="u2",
        display_name="B",
        knowledge_level="advanced",
        learning_style="step-by-step",
        interests=[],
        domains_of_focus=[],
    )

    history = [
        Feedback(user_id="u2", topic="law", rating=2, comment="too complex"),
        Feedback(user_id="u2", topic="law", rating=2, comment="still hard"),
        Feedback(user_id="u2", topic="law", rating=1, comment="too hard"),
    ]

    updated, updates = tune_profile_from_feedback(profile, history[-1], history)
    assert updated.knowledge_level == "intermediate"
    assert any("simplified after low ratings" in item for item in updates)

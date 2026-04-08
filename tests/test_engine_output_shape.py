from eilim.engine import EILIMEngine
from eilim.models import UserProfile


def test_output_shape_includes_required_sections():
    engine = EILIMEngine()
    profile = UserProfile(
        user_id="u2",
        display_name="Learner",
        knowledge_level="intermediate",
        learning_style="story",
        interests=["gaming"],
        domains_of_focus=["science"],
    )

    text = engine.explain(topic="photosynthesis", profile=profile, recent_topics=["plants"])

    assert "Topic: photosynthesis" in text
    assert "Quick take:" in text
    assert "Core explanation:" in text
    assert "Try it yourself:" in text
    assert "Story mode:" in text


def test_code_style_includes_code_block():
    engine = EILIMEngine()
    profile = UserProfile(
        user_id="u3",
        display_name="Coder",
        knowledge_level="beginner",
        learning_style="code",
        interests=["music"],
        domains_of_focus=[],
    )

    text = engine.explain(topic="variables", profile=profile, recent_topics=[])
    assert "```python" in text
    assert "def understand(topic):" in text


def test_gravity_topic_includes_specific_science_content():
    engine = EILIMEngine()
    profile = UserProfile(
        user_id="u4",
        display_name="Student",
        knowledge_level="beginner",
        learning_style="step-by-step",
        interests=[],
        domains_of_focus=["science"],
    )

    text = engine.explain(topic="teach me about gravity", profile=profile, recent_topics=[])

    assert "pull between things that have mass" in text
    assert "9.8 m/s^2" in text


def test_generic_prompt_uses_topic_direct_explanation_not_meta_learning_copy():
    engine = EILIMEngine()
    profile = UserProfile(
        user_id="u5",
        display_name="Learner",
        knowledge_level="beginner",
        learning_style="step-by-step",
        interests=[],
        domains_of_focus=[],
    )

    text = engine.explain(topic="teach me about inflation", profile=profile, recent_topics=[])

    assert "For inflation, start with a plain definition" in text
    assert "is just a way to answer a practical question" not in text

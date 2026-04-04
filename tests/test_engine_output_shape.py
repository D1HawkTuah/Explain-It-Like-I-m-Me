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

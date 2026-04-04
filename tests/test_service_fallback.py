from eilim.engine import EILIMEngine
from eilim.models import UserProfile
from eilim.service import generate_explanation


class LLMOk:
    enabled = True

    def __init__(self):
        self.last_semantic_context = None

    def explain(self, topic, profile, recent_topics, domain_hint, semantic_context=None):
        self.last_semantic_context = semantic_context
        return f"LLM answer for {topic}"


class LLMFails:
    enabled = True

    def explain(self, topic, profile, recent_topics, domain_hint):
        raise RuntimeError("network issue")


class LLMDisabled:
    enabled = False


def _profile() -> UserProfile:
    return UserProfile(
        user_id="u1",
        display_name="User One",
        knowledge_level="beginner",
        learning_style="step-by-step",
        interests=["music"],
        domains_of_focus=["math"],
    )


def test_uses_llm_when_available():
    engine = EILIMEngine()
    llm = LLMOk()
    semantic_context = {"summary": "user wants quick analogies"}
    explanation, domain, source = generate_explanation(
        topic="algebra basics",
        profile=_profile(),
        recent_topics=[],
        engine=engine,
        llm=llm,
        semantic_context=semantic_context,
    )

    assert source == "llm"
    assert explanation.startswith("LLM answer")
    assert domain == "school-math"
    assert llm.last_semantic_context == semantic_context


def test_falls_back_to_local_when_llm_fails():
    engine = EILIMEngine()
    explanation, domain, source = generate_explanation(
        topic="wifi setup",
        profile=_profile(),
        recent_topics=["router basics"],
        engine=engine,
        llm=LLMFails(),
    )

    assert source == "local"
    assert "Quick take:" in explanation
    assert domain == "tech-troubleshooting"


def test_uses_local_when_llm_disabled():
    engine = EILIMEngine()
    explanation, _, source = generate_explanation(
        topic="history of rome",
        profile=_profile(),
        recent_topics=[],
        engine=engine,
        llm=LLMDisabled(),
        semantic_context={"summary": "user prefers concise answers"},
    )

    assert source == "local"
    assert "Core explanation:" in explanation
    assert "Conversation memory:" in explanation

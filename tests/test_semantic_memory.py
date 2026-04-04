from eilim.semantic_memory import build_semantic_context


def test_build_semantic_context_extracts_summary_and_goals():
    history = [
        {"role": "user", "text": "Explain photosynthesis in simple terms."},
        {"role": "assistant", "text": "Photosynthesis is how plants turn sunlight into energy."},
        {"role": "user", "text": "Can you compare that with cellular respiration?"},
    ]

    context = build_semantic_context(history)

    assert context["turn_count"] == 3
    assert "Recent user intents:" in context["summary"]
    assert "Current goal:" in context["summary"]
    assert "cellular respiration" in context["latest_user_goal"].lower()

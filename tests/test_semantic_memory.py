from eilim.semantic_memory import build_semantic_context, retrieve_related_turns


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


def test_retrieve_related_turns_finds_semantically_similar_topics():
    history = [
        {"role": "user", "text": "Explain photosynthesis in simple terms."},
        {"role": "assistant", "text": "Photosynthesis converts sunlight into plant energy."},
        {"role": "user", "text": "Tell me about cellular respiration."},
    ]

    related = retrieve_related_turns(history, query="plants use sunlight to make energy", limit=2)

    assert related
    assert any("photosynthesis" in item["text"].lower() for item in related)

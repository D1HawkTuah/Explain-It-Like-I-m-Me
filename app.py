import os

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from eilim import (
    EILIMEngine,
    JSONStorage,
    LLMExplainer,
    UserProfile,
    build_semantic_context,
    generate_explanation,
    tune_profile_from_feedback,
)
from eilim.models import Feedback, Interaction

app = Flask(__name__)
app.secret_key = os.getenv("EILIM_FLASK_SECRET", "eilim-dev-secret")

storage = JSONStorage(root="data")
engine = EILIMEngine()
llm = LLMExplainer()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _chat_history() -> list[dict[str, str]]:
    history = session.get("chat_history")
    if not isinstance(history, list):
        history = []
    return history


def _set_chat_history(history: list[dict[str, str]]) -> None:
    # Keep session payload bounded so cookie-backed sessions remain lightweight.
    session["chat_history"] = history[-20:]


def _render_index(
    explanation: str | None,
    profile: UserProfile | None,
    source: str | None,
    updates: list[str],
    topic: str = "",
    user_id: str = "",
) -> str:
    semantic_context = build_semantic_context(_chat_history())
    return render_template(
        "index.html",
        explanation=explanation,
        profile=profile,
        source=source,
        topic=topic,
        user_id=user_id,
        updates=updates,
        chat_history=_chat_history(),
        semantic_summary=semantic_context.get("summary", ""),
    )


def _profile_from_form(user_id: str, existing: UserProfile | None) -> UserProfile:
    display_name = request.form.get("display_name", "").strip() or (existing.display_name if existing else user_id)
    knowledge_level = request.form.get("knowledge_level", "").strip().lower() or (
        existing.knowledge_level if existing else "beginner"
    )
    learning_style = request.form.get("learning_style", "").strip().lower() or (
        existing.learning_style if existing else "step-by-step"
    )
    interests = _parse_csv(request.form.get("interests", ""))
    domains = _parse_csv(request.form.get("domains_of_focus", ""))
    self_sample = request.form.get("self_explainer_sample", "").strip()
    survey = request.form.get("onboarding_survey", "").strip().lower()
    quiz_raw = request.form.get("calibration_quiz_score", "").strip()

    if existing and not interests:
        interests = existing.interests
    if existing and not domains:
        domains = existing.domains_of_focus

    quiz_score = existing.calibration_quiz_score if existing else -1
    if quiz_raw:
        try:
            quiz_score = max(0, min(3, int(quiz_raw)))
        except ValueError:
            pass

    if knowledge_level not in {"beginner", "intermediate", "advanced"}:
        if quiz_score >= 3:
            knowledge_level = "advanced"
        elif quiz_score == 2:
            knowledge_level = "intermediate"
        else:
            knowledge_level = "beginner"

    return UserProfile(
        user_id=user_id,
        display_name=display_name,
        knowledge_level=knowledge_level,
        learning_style=learning_style,
        interests=interests,
        domains_of_focus=domains,
        self_explainer_sample=self_sample or (existing.self_explainer_sample if existing else ""),
        onboarding_survey=survey or (existing.onboarding_survey if existing else ""),
        calibration_quiz_score=quiz_score,
    )


@app.get("/")
def index():
    return _render_index(explanation=None, profile=None, source=None, updates=[])


@app.post("/chat/clear")
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("index"))


@app.get("/memory/inspect")
def inspect_memory():
    history = _chat_history()
    semantic_context = build_semantic_context(history)
    return render_template(
        "memory_inspector.html",
        chat_history=history,
        semantic_context=semantic_context,
        session_turn_count=len(history),
    )


@app.get("/memory/inspect.json")
def inspect_memory_json():
    history = _chat_history()
    semantic_context = build_semantic_context(history)
    return jsonify(
        {
            "chat_history": history,
            "semantic_context": semantic_context,
            "session_turn_count": len(history),
        }
    )


@app.post("/explain")
def explain_topic():
    user_id = request.form.get("user_id", "").strip() or "guest"
    topic = request.form.get("topic", "").strip()

    profile_existing = storage.load_profile(user_id)
    profile = _profile_from_form(user_id=user_id, existing=profile_existing)
    storage.save_profile(profile)

    explanation = None
    source = None
    updates: list[str] = []
    history = _chat_history()

    if topic:
        semantic_context = build_semantic_context(history)

        recent_topics = storage.recent_topics(user_id=user_id, limit=5)
        explanation, domain, source = generate_explanation(
            topic=topic,
            profile=profile,
            recent_topics=recent_topics,
            engine=engine,
            llm=llm,
            semantic_context=semantic_context,
        )

        storage.save_interaction(
            Interaction(
                user_id=user_id,
                topic=topic,
                explanation=explanation,
                domain=domain,
            )
        )

        history.append({"role": "user", "text": topic})
        history.append({"role": "assistant", "text": explanation, "source": source or "local"})
        _set_chat_history(history)

    return _render_index(
        explanation=explanation,
        profile=profile,
        source=source,
        updates=updates,
        topic=topic,
        user_id=user_id,
    )


@app.post("/feedback")
def save_feedback():
    user_id = request.form.get("user_id", "").strip() or "guest"
    topic = request.form.get("topic", "").strip() or "general"
    comment = request.form.get("comment", "").strip()
    rating_raw = request.form.get("rating", "").strip()

    profile = storage.load_profile(user_id)
    if not profile:
        profile = UserProfile(
            user_id=user_id,
            display_name=user_id,
            knowledge_level="beginner",
            learning_style="step-by-step",
            interests=[],
            domains_of_focus=[],
        )

    try:
        rating = max(1, min(5, int(rating_raw)))
    except ValueError:
        return redirect(url_for("index"))

    storage.save_feedback(Feedback(user_id=user_id, topic=topic, rating=rating, comment=comment))
    history = storage.recent_feedback(user_id=user_id, limit=5)
    latest = history[-1]
    profile, updates = tune_profile_from_feedback(
        profile=profile,
        latest_feedback=latest,
        recent_feedback=history,
    )
    storage.save_profile(profile)

    return _render_index(
        explanation="Feedback saved. Ask another topic to see the tuned behavior.",
        profile=profile,
        source="feedback",
        updates=updates,
        topic=topic,
        user_id=user_id,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

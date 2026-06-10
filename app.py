import logging
import os
import time
from functools import wraps

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
from eilim.validation import (
    InputValidationError,
    normalize_user_id,
    parse_csv_field,
    validate_comment,
    validate_display_name,
    validate_knowledge_level,
    validate_learning_style,
    validate_quiz_score,
    validate_rating,
    validate_request_payload_size,
    validate_survey_preference,
    validate_self_explainer,
    validate_topic,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----- Fallback generator (same as CLI) -----
def local_fallback_explanation(
    topic: str,
    profile: UserProfile,
    recent_topics: list[str] | None = None,
) -> str:
    style = profile.learning_style or "step-by-step"
    level = profile.knowledge_level or "beginner"
    interests = profile.interests or []
    domains = profile.domains_of_focus or []
    self_sample = profile.self_explainer_sample or ""

    depth = {"beginner": "simple", "intermediate": "moderate", "advanced": "in-depth"}
    depth_term = depth.get(level, "simple")

    interest_hook = ""
    if interests:
        interest_hook = f" Think of it like how {interests[0].capitalize()} might approach this problem."

    opener_map = {
        "step-by-step": f"Let's break down **{topic}** into manageable steps.",
        "visual": f"Picture **{topic}** as a visual scene…",
        "story": f"Here's a story that will help you understand **{topic}**.",
        "code": f"Let's explore **{topic}** through some code examples.",
    }
    opener = opener_map.get(style, f"Let's explore **{topic}** together.")

    domain_note = ""
    if domains:
        domain_note = f" Because you're interested in {', '.join(domains)}, we'll focus on how {topic} applies there."

    self_hint = ""
    if self_sample:
        first_sentence = self_sample.split(".")[0].strip()
        if first_sentence:
            self_hint = f" As you once put it: \"{first_sentence}\"."

    explanation = (
        f"{opener}\n\n"
        f"This is a **{depth_term}** explanation tailored for a {level} learner."
        f"{interest_hook}{domain_note}\n\n"
        f"### Key Points about {topic}:\n"
        f"1. **Core Idea**: The concept of {topic} is central because it helps us understand related fields.\n"
        f"2. **Why it matters**: Knowing {topic} can improve your everyday decision-making and problem-solving.\n"
        f"3. **How to think about it**: Start by asking \"What does {topic} mean in my own words?\"{self_hint}\n"
        f"4. **Practical takeaway**: Try applying {topic} to a situation you encountered recently.\n\n"
        f"Would you like me to go deeper or switch to a different style?"
    )

    if recent_topics:
        last_topic = recent_topics[-1] if recent_topics else ""
        if last_topic:
            explanation += f"\n> 🔗 You recently asked about **{last_topic}** — this connects nicely because both are about understanding complex ideas."

    return explanation


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "explain-it-secret")

storage = JSONStorage(root="data")
engine = EILIMEngine()
llm = LLMExplainer()


def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        now = time.time()
        window = 60
        max_requests = 30

        history = session.get("_rate_limit", [])
        if not isinstance(history, list):
            history = []

        history = [timestamp for timestamp in history if now - timestamp < window]
        if len(history) >= max_requests:
            return jsonify({"error": "Too many requests"}), 429

        history.append(now)
        session["_rate_limit"] = history
        return func(*args, **kwargs)

    return wrapper


def _profile_from_form(user_id: str, existing: UserProfile | None = None) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        display_name=validate_display_name(
            request.form.get("display_name", ""), fallback=user_id
        ),
        knowledge_level=validate_knowledge_level(
            request.form.get(
                "knowledge_level",
                existing.knowledge_level if existing else "beginner",
            )
        ),
        learning_style=validate_learning_style(
            request.form.get(
                "learning_style",
                existing.learning_style if existing else "step-by-step",
            )
        ),
        interests=parse_csv_field(request.form.get("interests", "")),
        domains_of_focus=parse_csv_field(request.form.get("domains_of_focus", "")),
        self_explainer_sample=validate_self_explainer(
            request.form.get(
                "self_explainer_sample",
                existing.self_explainer_sample if existing else "",
            )
        ),
        onboarding_survey=validate_survey_preference(
            request.form.get(
                "onboarding_survey",
                existing.onboarding_survey if existing else "",
            )
        ),
        calibration_quiz_score=validate_quiz_score(
            request.form.get("calibration_quiz_score", "")
        ),
    )


def _chat_history() -> list[dict[str, str]]:
    return session.get("chat_history", [])


def _set_chat_history(history: list[dict[str, str]]) -> None:
    session["chat_history"] = history


def _render_index(
    explanation: str | None,
    profile: UserProfile | None,
    source: str | None,
    updates: list[str],
    topic: str,
    user_id: str,
) -> str:
    history = _chat_history()
    semantic_context = build_semantic_context(history)
    semantic_summary = semantic_context.get("summary", "")

    return render_template(
        "index.html",
        explanation=explanation,
        profile=profile,
        source=source,
        updates=updates,
        topic=topic,
        user_id=user_id,
        semantic_summary=semantic_summary,
    )


@app.errorhandler(InputValidationError)
def _handle_validation_error(error):
    return jsonify({"error": str(error)}), 400


@app.get("/")
def index():
    return _render_index(
        explanation=None,
        profile=None,
        source=None,
        updates=[],
        topic="",
        user_id="guest",
    )


@app.post("/explain")
@rate_limited
def explain_topic():
    validate_request_payload_size(dict(request.form))

    user_id = normalize_user_id(request.form.get("user_id", ""))
    topic = validate_topic(request.form.get("topic", ""))

    profile_existing = storage.load_profile(user_id)
    profile = _profile_from_form(user_id=user_id, existing=profile_existing)
    storage.save_profile(profile)

    explanation = None
    source = "local-fallback"
    domain = "general"
    updates: list[str] = []
    history = _chat_history()

    if topic:
        semantic_context = build_semantic_context(history)
        recent_topics = storage.recent_topics(user_id=user_id, limit=5)

        try:
            explanation, domain, source = generate_explanation(
                topic=topic,
                profile=profile,
                recent_topics=recent_topics,
                engine=engine,
                llm=llm,
                semantic_context=semantic_context,
            )
        except Exception as e:
            logger.warning("Engine/LLM failed: %s. Using fallback.", e)
            explanation = None

        if not explanation:
            explanation = local_fallback_explanation(topic, profile, recent_topics)
            source = "local-fallback"

        storage.save_interaction(
            Interaction(
                user_id=user_id,
                topic=topic,
                explanation=explanation,
                domain=domain,
            )
        )

        history.append({"role": "user", "text": topic})
        history.append({"role": "assistant", "text": explanation, "source": source})
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
@rate_limited
def submit_feedback():
    validate_request_payload_size(dict(request.form))

    user_id = normalize_user_id(request.form.get("user_id", ""))
    topic = validate_topic(request.form.get("topic", ""))
    rating = validate_rating(request.form.get("rating", ""))
    comment = validate_comment(request.form.get("comment", ""))

    profile = storage.load_profile(user_id)
    if profile is None:
        profile = UserProfile(
            user_id=user_id,
            display_name=user_id,
            knowledge_level="beginner",
            learning_style="step-by-step",
        )

    if rating >= 1:
        storage.save_feedback(
            Feedback(user_id=user_id, topic=topic, rating=rating, comment=comment)
        )
        history = storage.recent_feedback(user_id=user_id, limit=5)
        if history:
            latest = history[-1]
            profile, updates = tune_profile_from_feedback(
                profile=profile,
                latest_feedback=latest,
                recent_feedback=history,
            )
            storage.save_profile(profile)
        else:
            updates = []
    else:
        updates = []

    return redirect(url_for("index"))


@app.get("/memory/inspect.json")
def memory_inspect_json():
    history = _chat_history()
    semantic_context = build_semantic_context(history)

    return jsonify(
        chat_history=history,
        semantic_context=semantic_context,
        session_turn_count=len(history),
    )


@app.get("/memory/inspect")
def memory_inspect():
    history = _chat_history()
    semantic_context = build_semantic_context(history)

    return render_template(
        "memory_inspector.html",
        chat_history=history,
        semantic_context=semantic_context,
        session_turn_count=len(history),
    )


@app.post("/chat/clear")
def clear_chat_history():
    session.pop("chat_history", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

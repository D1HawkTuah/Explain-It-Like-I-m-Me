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
    normalize_user_id,
    parse_csv_field,
    validate_comment,
    validate_display_name,
    validate_knowledge_level,
    validate_learning_style,
    validate_quiz_score,
    validate_rating,
    validate_survey_preference,
    validate_topic,
    validate_self_explainer,
    InputValidationError,
    validate_request_payload_size,
)

logger = logging.getLogger(__name__)

# Flask app initialization with security checks
app = Flask(__name__)

# Fail-fast on missing Flask secret in production environments
flask_secret = os.getenv("EILIM_FLASK_SECRET")
if not flask_secret:
    env_mode = os.getenv("FLASK_ENV", "development")
    if env_mode != "development":
        raise RuntimeError(
            "EILIM_FLASK_SECRET environment variable must be set in production. "
            "Set FLASK_ENV=development to use dev mode."
        )
    flask_secret = "eilim-dev-secret"
    logger.warning("Using development Flask secret key. Set EILIM_FLASK_SECRET in production.")

app.secret_key = flask_secret

# Request size limits
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request body
app.config["JSON_SORT_KEYS"] = False

storage = JSONStorage(root="data")
engine = EILIMEngine()
llm = LLMExplainer()

# Simple in-memory rate limit tracking (per user_id)
_rate_limit_tracker: dict[str, tuple[float, int]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # requests per window


def _check_rate_limit(user_id: str) -> bool:
    """Check if user_id has exceeded rate limit. Returns True if OK, False if limited."""
    now = time.time()
    if user_id not in _rate_limit_tracker:
        _rate_limit_tracker[user_id] = (now, 1)
        return True
    
    window_start, count = _rate_limit_tracker[user_id]
    if now - window_start > RATE_LIMIT_WINDOW:
        _rate_limit_tracker[user_id] = (now, 1)
        return True
    
    if count >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    _rate_limit_tracker[user_id] = (window_start, count + 1)
    return True


def rate_limited(f):
    """Decorator to enforce rate limiting per user_id in form data."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_id = normalize_user_id(request.form.get("user_id", ""))
        if not _check_rate_limit(user_id):
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            return jsonify({"error": "Too many requests"}), 429
        return f(*args, **kwargs)
    return decorated


@app.before_request
def validate_request_payload():
    """Validate request payload size and structure."""
    if request.method in ("POST", "PUT"):
        try:
            if request.form:
                validate_request_payload_size(dict(request.form), max_fields=20)
        except InputValidationError as e:
            logger.warning(f"Invalid request payload: {str(e)}")
            return jsonify({"error": str(e)}), 400


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
    display_name_raw = request.form.get("display_name", "").strip()
    display_name = validate_display_name(
        display_name_raw or (existing.display_name if existing else None),
        user_id
    )
    
    knowledge_level = validate_knowledge_level(
        request.form.get("knowledge_level", "").strip() or (existing.knowledge_level if existing else None)
    )
    
    learning_style = validate_learning_style(
        request.form.get("learning_style", "").strip() or (existing.learning_style if existing else None)
    )
    
    interests = parse_csv_field(request.form.get("interests", ""))
    if existing and not interests:
        interests = existing.interests
    
    domains = parse_csv_field(request.form.get("domains_of_focus", ""))
    if existing and not domains:
        domains = existing.domains_of_focus

    self_sample = validate_self_explainer(request.form.get("self_explainer_sample", ""))
    if existing and not self_sample:
        self_sample = existing.self_explainer_sample

    survey = validate_survey_preference(
        request.form.get("onboarding_survey", "").strip() or (existing.onboarding_survey if existing else None)
    )

    quiz_score = validate_quiz_score(request.form.get("calibration_quiz_score", "").strip())
    if existing and quiz_score == -1:
        quiz_score = existing.calibration_quiz_score

    return UserProfile(
        user_id=user_id,
        display_name=display_name,
        knowledge_level=knowledge_level,
        learning_style=learning_style,
        interests=interests,
        domains_of_focus=domains,
        self_explainer_sample=self_sample,
        onboarding_survey=survey,
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
@rate_limited
def explain_topic():
    user_id = normalize_user_id(request.form.get("user_id", ""))
    topic = validate_topic(request.form.get("topic", ""))

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
@rate_limited
def save_feedback():
    user_id = normalize_user_id(request.form.get("user_id", ""))
    topic = validate_topic(request.form.get("topic", "") or "general")
    comment = validate_comment(request.form.get("comment", ""))
    rating = validate_rating(request.form.get("rating", "").strip())

    if rating < 1:
        return redirect(url_for("index"))

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

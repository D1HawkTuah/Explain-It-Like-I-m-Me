import asyncio
import io
import logging
import os
import time
from functools import wraps

import edge_tts
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file

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


def _session_chat_history() -> list[dict[str, str]]:
    return session.get("chat_history", [])


def _set_session_chat_history(history: list[dict[str, str]]) -> None:
    session["chat_history"] = history


def _load_conversation_history(user_id: str) -> list[dict[str, str]]:
    history = storage.conversation_history(user_id=user_id, limit=50)
    if history:
        return history
    return _session_chat_history()


def _profile_from_payload(user_id: str, payload: dict, existing: UserProfile | None = None) -> UserProfile:
    display_name = validate_display_name(
        payload.get("display_name")
        or (existing.display_name if existing else user_id),
        fallback=user_id,
    )
    knowledge_level = validate_knowledge_level(
        payload.get("knowledge_level")
        or (existing.knowledge_level if existing else "beginner")
    )
    learning_style = validate_learning_style(
        payload.get("learning_style")
        or (existing.learning_style if existing else "step-by-step")
    )
    if "interests" in payload:
        interests = parse_csv_field(payload.get("interests", ""))
    else:
        interests = existing.interests if existing else []
    if "domains_of_focus" in payload:
        domains_of_focus = parse_csv_field(payload.get("domains_of_focus", ""))
    else:
        domains_of_focus = existing.domains_of_focus if existing else []
    if "self_explainer_sample" in payload:
        self_explainer_sample = validate_self_explainer(payload.get("self_explainer_sample", ""))
    else:
        self_explainer_sample = existing.self_explainer_sample if existing else ""
    if "onboarding_survey" in payload:
        onboarding_survey = validate_survey_preference(payload.get("onboarding_survey", ""))
    else:
        onboarding_survey = existing.onboarding_survey if existing else ""
    if "calibration_quiz_score" in payload:
        calibration_quiz_score = validate_quiz_score(payload.get("calibration_quiz_score", ""))
    else:
        calibration_quiz_score = existing.calibration_quiz_score if existing else -1

    return UserProfile(
        user_id=user_id,
        display_name=display_name,
        knowledge_level=knowledge_level,
        learning_style=learning_style,
        interests=interests,
        domains_of_focus=domains_of_focus,
        self_explainer_sample=self_explainer_sample,
        onboarding_survey=onboarding_survey,
        calibration_quiz_score=calibration_quiz_score,
    )


def _render_index(
    explanation: str | None,
    profile: UserProfile | None,
    source: str | None,
    updates: list[str],
    topic: str,
    user_id: str,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    history = conversation if conversation is not None else _load_conversation_history(user_id)
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
        chat_history=history,
    )


def _append_chat_turns(user_id: str, history: list[dict[str, str]], user_text: str, assistant_text: str, source: str) -> list[dict[str, str]]:
    storage.save_conversation_turn(user_id=user_id, role="user", text=user_text, source="")
    storage.save_conversation_turn(user_id=user_id, role="assistant", text=assistant_text, source=source)
    new_history = history + [
        {"role": "user", "text": user_text},
        {"role": "assistant", "text": assistant_text, "source": source},
    ]
    _set_session_chat_history(new_history)
    return new_history


def _execute_explanation(user_id: str, profile: UserProfile, topic: str) -> tuple[str, str, list[dict[str, str]], list[str]]:
    history = _load_conversation_history(user_id)
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
        source = "local-fallback"

    if not explanation:
        explanation = local_fallback_explanation(topic, profile, recent_topics)
        source = "local-fallback"
        domain = "general"

    storage.save_interaction(
        Interaction(
            user_id=user_id,
            topic=topic,
            explanation=explanation,
            domain=domain,
        )
    )

    updated_history = _append_chat_turns(user_id, history, topic, explanation, source)
    return explanation, source, updated_history, []


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
    payload = request.get_json(silent=True) or request.form.to_dict()
    validate_request_payload_size(payload)

    user_id = normalize_user_id(payload.get("user_id", ""))
    topic = validate_topic(payload.get("topic", ""))

    profile_existing = storage.load_profile(user_id)
    profile = _profile_from_payload(user_id=user_id, payload=payload, existing=profile_existing)
    storage.save_profile(profile)

    explanation = None
    source = "local-fallback"
    updates: list[str] = []
    conversation = _load_conversation_history(user_id)

    if topic:
        explanation, source, conversation, updates = _execute_explanation(user_id, profile, topic)

    if request.is_json:
        return jsonify(
            explanation=explanation,
            source=source,
            profile=profile.to_dict(),
            updates=updates,
            topic=topic,
            user_id=user_id,
            conversation=conversation,
        )

    return _render_index(
        explanation=explanation,
        profile=profile,
        source=source,
        updates=updates,
        topic=topic,
        user_id=user_id,
        conversation=conversation,
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


def _conversation_history_for_inspector() -> list[dict[str, str]]:
    user_id = normalize_user_id(request.args.get("user_id", ""))
    session_history = _session_chat_history()
    if session_history:
        return session_history
    return _load_conversation_history(user_id)


@app.get("/memory/inspect.json")
def memory_inspect_json():
    history = _conversation_history_for_inspector()
    semantic_context = build_semantic_context(history)

    return jsonify(
        chat_history=history,
        semantic_context=semantic_context,
        session_turn_count=len(history),
    )


@app.get("/memory/inspect")
def memory_inspect():
    history = _conversation_history_for_inspector()
    semantic_context = build_semantic_context(history)

    return render_template(
        "memory_inspector.html",
        chat_history=history,
        semantic_context=semantic_context,
        session_turn_count=len(history),
    )


@app.post("/chat/clear")
def clear_chat_history():
    user_id = normalize_user_id(request.form.get("user_id", "guest"))
    storage.clear_conversation_history(user_id=user_id)
    session.pop("chat_history", None)
    return redirect(url_for("index"))


@app.post("/tts")
@rate_limited
def generate_tts():
    """Generate text-to-speech audio from text."""
    payload = request.get_json(silent=True) or request.form.to_dict()
    text = (payload.get("text") or "").strip()
    voice = (payload.get("voice") or "en-US-AriaNeural").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text too long (max 5000 chars)"}), 400

    try:
        async def _generate_audio():
            communicate = edge_tts.Communicate(text, voice=voice)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)
            return audio_buffer.getvalue()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(_generate_audio())
        loop.close()

        audio_io = io.BytesIO(audio_bytes)
        return send_file(
            audio_io,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="explanation.mp3",
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)[:100]}")
        return jsonify({"error": "Failed to generate audio"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

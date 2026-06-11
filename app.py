import asyncio
import datetime
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


def infer_learning_style_from_text(text: str) -> str:
    normalized = (text or "").lower()
    if any(keyword in normalized for keyword in ["visual", "diagram", "picture", "map", "chart", "graph"]):
        return "visual"
    if any(keyword in normalized for keyword in ["story", "narrative", "imagine", "scene", "character", "tale"]):
        return "story"
    if any(keyword in normalized for keyword in ["code", "example", "program", "script", "algorithm", "function"]):
        return "code"
    return "step-by-step"


def infer_knowledge_level_from_text(text: str) -> str:
    normalized = (text or "").lower()
    if any(keyword in normalized for keyword in ["advanced", "deep", "detailed", "in-depth", "complex"]):
        return "advanced"
    if any(keyword in normalized for keyword in ["intermediate", "more than basics", "some experience", "familiar"]):
        return "intermediate"
    return "beginner"


def _quiz_count_from_payload(payload: dict) -> int:
    try:
        count = int(payload.get("count", 3))
    except (TypeError, ValueError):
        count = 3
    return max(1, min(5, count))


def _parse_boolean(value: object | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _wrap_text_lines(text: str, width: int = 95) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = words[0]
        for word in words[1:]:
            if len(current) + len(word) + 1 > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}"
        lines.append(current)
    return lines


def _escape_pdf_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped


def _build_summary_markdown(
    user_id: str,
    profile: UserProfile | None,
    history: list[dict[str, str]],
    semantic_context: dict[str, str],
    mastery_summary: dict[str, object],
) -> str:
    display_name = profile.display_name if profile and profile.display_name else user_id
    lines: list[str] = [f"# Learning Summary for {display_name}", ""]
    lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Profile")
    lines.append(f"- User ID: {user_id}")
    lines.append(f"- Display name: {display_name}")
    if profile:
        lines.append(f"- Knowledge level: {profile.knowledge_level or 'beginner'}")
        lines.append(f"- Learning style: {profile.learning_style or 'step-by-step'}")
        lines.append(f"- Interests: {', '.join(profile.interests or []) or 'None'}")
        lines.append(f"- Domains of focus: {', '.join(profile.domains_of_focus or []) or 'None'}")
        if profile.self_explainer_sample:
            lines.append(f"- Self-explainer sample: {profile.self_explainer_sample}")
    lines.append("")
    lines.append("## Mastery Summary")
    lines.append(f"- Tracked topics: {mastery_summary.get('tracked_topics', 0)}")
    lines.append(f"- Due count: {mastery_summary.get('due_count', 0)}")
    due_topics = mastery_summary.get('due_topics', []) or []
    lines.append(f"- Due topics: {', '.join(due_topics) if due_topics else 'None'}")
    next_due = mastery_summary.get('next_due_days')
    lines.append(f"- Next review in: {next_due if next_due is not None else 'N/A'} days")
    lines.append("")
    lines.append("## Semantic Summary")
    semantic_summary = semantic_context.get('summary', '')
    lines.append(semantic_summary or "No semantic summary available.")
    lines.append("")
    lines.append("## Chat History")
    if history:
        for turn in history:
            role = turn.get('role', 'user').capitalize()
            text = turn.get('text', '').strip()
            source = turn.get('source', '')
            if source:
                lines.append(f"- **{role}** ({source}): {text}")
            else:
                lines.append(f"- **{role}**: {text}")
    else:
        lines.append("No chat history available.")
    return "\n".join(lines)


def _build_pdf_from_markdown(markdown_text: str) -> io.BytesIO:
    lines: list[str] = []
    for paragraph in markdown_text.splitlines():
        wrapped = _wrap_text_lines(paragraph, width=90)
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")

    lines_per_page = 55
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    if not pages:
        pages = [[""]]

    pdf_parts: list[bytes] = []
    pdf_parts.append(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    object_index = 1

    def add_object(content: bytes) -> None:
        nonlocal pdf_parts, offsets, object_index
        offsets.append(sum(len(part) for part in pdf_parts))
        pdf_parts.append(f"{object_index} 0 obj\n".encode('latin1'))
        pdf_parts.append(content)
        pdf_parts.append(b"endobj\n")
        object_index += 1

    # Catalog and Pages placeholder
    add_object(b"<< /Type /Catalog /Pages 2 0 R >>\n")
    add_object(b"<< /Type /Pages /Kids [" + b" ".join([f"{4 + i} 0 R".encode('latin1') for i in range(len(pages))]) + b"] /Count %d >>\n" % len(pages))
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\n")

    page_ids = []
    content_ids = []
    for page_lines in pages:
        page_ids.append(object_index)
        add_object(b"" )
    for _ in pages:
        content_ids.append(object_index)
        add_object(b"" )

    for idx, page_lines in enumerate(pages):
        content_stream_lines: list[str] = ["BT", "/F1 10 Tf", "50 760 Td"]
        for line_no, line in enumerate(page_lines):
            escaped = _escape_pdf_text(line)
            if line_no == len(page_lines) - 1:
                content_stream_lines.append(f"({escaped}) Tj")
            else:
                content_stream_lines.append(f"({escaped}) Tj")
                content_stream_lines.append("T*")
        content_stream_lines.append("ET")
        content_stream = "\n".join(content_stream_lines).encode('latin1', errors='replace')
        stream = b"<< /Length %d >>\nstream\n" % len(content_stream) + content_stream + b"\nendstream\n"
        pdf_parts[content_ids[idx] * 2 - 1] = stream

    for idx, page_id in enumerate(page_ids):
        page_content = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>\n" % content_ids[idx]
        )
        pdf_parts[page_id * 2 - 1] = page_content

    # Build xref
    xref_start = sum(len(part) for part in pdf_parts)
    pdf_parts.append(b"xref\n0 %d\n0000000000 65535 f \n" % (object_index,))
    for offset in offsets:
        pdf_parts.append(f"{offset:010d} 00000 n \n".encode('latin1'))
    pdf_parts.append(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (object_index, xref_start))

    buffer = io.BytesIO(b"".join(pdf_parts))
    buffer.seek(0)
    return buffer


def _profile_from_payload(user_id: str, payload: dict, existing: UserProfile | None = None) -> UserProfile:
    display_name = validate_display_name(
        payload.get("display_name")
        or (existing.display_name if existing else user_id),
        fallback=user_id,
    )
    self_explainer_sample = (
        validate_self_explainer(payload.get("self_explainer_sample", ""))
        if "self_explainer_sample" in payload
        else existing.self_explainer_sample if existing else ""
    )
    suggested_style = infer_learning_style_from_text(self_explainer_sample)
    suggested_level = infer_knowledge_level_from_text(self_explainer_sample)
    knowledge_level = validate_knowledge_level(
        payload.get("knowledge_level")
        or (existing.knowledge_level if existing else suggested_level)
    )
    learning_style = validate_learning_style(
        payload.get("learning_style")
        or (existing.learning_style if existing else suggested_style)
    )
    if "interests" in payload:
        interests = parse_csv_field(payload.get("interests", ""))
    else:
        interests = existing.interests if existing else []
    if "domains_of_focus" in payload:
        domains_of_focus = parse_csv_field(payload.get("domains_of_focus", ""))
    else:
        domains_of_focus = existing.domains_of_focus if existing else []
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
    show_profile_wizard = profile is None or not getattr(profile, "self_explainer_sample", "")
    mastery_summary = storage.mastery_overview(user_id=user_id)

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
        show_profile_wizard=show_profile_wizard,
        mastery_summary=mastery_summary,
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


def _execute_explanation(
    user_id: str,
    profile: UserProfile,
    topic: str,
    variant: bool = False,
) -> tuple[str, str, list[dict[str, str]], list[str]]:
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
            variant=variant,
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
    variant = _parse_boolean(payload.get("variant", False))

    profile_existing = storage.load_profile(user_id)
    profile = _profile_from_payload(user_id=user_id, payload=payload, existing=profile_existing)
    storage.save_profile(profile)

    explanation = None
    source = "local-fallback"
    updates: list[str] = []
    conversation = _load_conversation_history(user_id)

    if topic:
        explanation, source, conversation, updates = _execute_explanation(user_id, profile, topic, variant=variant)

    if request.is_json:
        mastery_summary = storage.mastery_overview(user_id=user_id)
        return jsonify(
            explanation=explanation,
            source=source,
            profile=profile.to_dict(),
            updates=updates,
            topic=topic,
            user_id=user_id,
            conversation=conversation,
            mastery_summary=mastery_summary,
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


@app.post("/quiz")
@rate_limited
def quiz_topic():
    payload = request.get_json(silent=True) or request.form.to_dict()
    validate_request_payload_size(payload)

    user_id = normalize_user_id(payload.get("user_id", ""))
    topic = validate_topic(payload.get("topic", ""))
    if not topic:
        raise InputValidationError("Quiz topic is required.")
    question_count = _quiz_count_from_payload(payload)

    profile_existing = storage.load_profile(user_id)
    profile = _profile_from_payload(user_id=user_id, payload=payload, existing=profile_existing)
    storage.save_profile(profile)

    quiz_questions = engine.generate_quiz(
        topic=topic,
        profile=profile,
        recent_topics=storage.recent_topics(user_id=user_id, limit=5),
        semantic_context=build_semantic_context(_load_conversation_history(user_id)),
        count=question_count,
    )

    if request.is_json:
        mastery_summary = storage.mastery_overview(user_id=user_id)
        return jsonify(
            topic=topic,
            user_id=user_id,
            quiz=quiz_questions,
            mastery_summary=mastery_summary,
        )

    return redirect(url_for("index"))


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
        mastery_record = storage.update_mastery_for_feedback(user_id=user_id, topic=topic, rating=rating)
        history = storage.recent_feedback(user_id=user_id, limit=5)
        if history:
            latest = history[-1]
            profile, updates = tune_profile_from_feedback(
                profile=profile,
                latest_feedback=latest,
                recent_feedback=history,
            )
            storage.save_profile(profile)
            updates.append(
                f"mastery for '{topic}' updated: score={mastery_record.mastery_score}, next review in {mastery_record.interval_days} days"
            )
        else:
            updates = [
                f"mastery for '{topic}' recorded: score={mastery_record.mastery_score}, next review in {mastery_record.interval_days} days"
            ]
    else:
        updates = []

    return redirect(url_for("index"))


@app.get("/export/summary")
def export_summary():
    user_id = normalize_user_id(request.args.get("user_id", "guest"))
    export_format = (request.args.get("format", "markdown") or "markdown").lower()

    profile = storage.load_profile(user_id)
    history = _load_conversation_history(user_id)
    semantic_context = build_semantic_context(history)
    mastery_summary = storage.mastery_overview(user_id=user_id)
    markdown = _build_summary_markdown(
        user_id=user_id,
        profile=profile,
        history=history,
        semantic_context=semantic_context,
        mastery_summary=mastery_summary,
    )

    if export_format == "pdf":
        file_buffer = _build_pdf_from_markdown(markdown)
        file_name = f"{user_id or 'guest'}-learning-summary.pdf"
        return send_file(
            file_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=file_name,
        )

    if export_format == "markdown":
        file_buffer = io.BytesIO(markdown.encode("utf-8"))
        file_name = f"{user_id or 'guest'}-learning-summary.md"
        return send_file(
            file_buffer,
            mimetype="text/markdown; charset=utf-8",
            as_attachment=True,
            download_name=file_name,
        )

    return jsonify({"error": "Unsupported export format."}), 400


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

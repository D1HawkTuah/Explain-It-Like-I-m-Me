import logging
from eilim import (
    EILIMEngine,
    JSONStorage,
    LLMExplainer,
    UserProfile,
    generate_explanation,
    tune_profile_from_feedback,
)
from eilim.models import Feedback, Interaction
from eilim.validation import (
    normalize_user_id,
    parse_csv_field,
    validate_display_name,
    validate_knowledge_level,
    validate_learning_style,
    validate_quiz_score,
    validate_rating,
    validate_survey_preference,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Fallback explanation generator – always returns a result
# ----------------------------------------------------------------------
def local_fallback_explanation(
    topic: str,
    profile: UserProfile,
    recent_topics: list[str] | None = None,
) -> str:
    """Generate a personalised explanation using only the profile info.
    This runs completely offline and never fails."""
    style = profile.learning_style or "step-by-step"
    level = profile.knowledge_level or "beginner"
    interests = profile.interests or []
    domains = profile.domains_of_focus or []
    survey = profile.onboarding_survey or "examples-first"
    self_sample = profile.self_explainer_sample or ""

    # Determine complexity
    depth = {"beginner": "simple", "intermediate": "moderate", "advanced": "in-depth"}
    depth_term = depth.get(level, "simple")

    # Build a hook from interests
    interest_hook = ""
    if interests:
        # pick first interest to relate the topic
        interest_word = interests[0].capitalize()
        interest_hook = f" Think of it like how {interest_word} might approach this problem."

    # Style-specific opening
    opener_map = {
        "step-by-step": f"Let's break down **{topic}** into manageable steps.",
        "visual": f"Picture **{topic}** as a visual scene…",
        "story": f"Here's a story that will help you understand **{topic}**.",
        "code": f"Let's explore **{topic}** through some code examples.",
    }
    opener = opener_map.get(style, f"Let's explore **{topic}** together.")

    # Domain emphasis
    domain_note = ""
    if domains:
        domain_note = f" Because you're interested in {', '.join(domains)}, we'll focus on how {topic} applies there."

    # Incorporate self-explainer phrasing if available
    self_hint = ""
    if self_sample:
        # Use first sentence as a mirroring phrase
        first_sentence = self_sample.split(".")[0].strip()
        if first_sentence:
            self_hint = f" As you once put it: \"{first_sentence}\"."

    # Construct the explanation
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

    # If recent topics exist, mention continuity
    if recent_topics:
        last_topic = recent_topics[-1] if recent_topics else ""
        if last_topic:
            explanation += f"\n> 🔗 You recently asked about **{last_topic}** — this connects nicely because both are about understanding complex ideas."

    return explanation


# CLI helpers (unchanged, but included for completeness)
def ask(prompt: str, default: str = "") -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        return default
    return value if value else default


def create_profile(user_id: str) -> UserProfile:
    display_name = validate_display_name(
        ask("Display name (optional): ", ""),
        fallback=user_id,
    )

    knowledge_level = validate_knowledge_level(
        ask("Knowledge level [beginner/intermediate/advanced] (beginner): ", "beginner")
    )

    learning_style = validate_learning_style(
        ask("Learning style [step-by-step/visual/story/code] (step-by-step): ", "step-by-step")
    )

    interests = parse_csv_field(
        ask("Interests (comma-separated, optional): ", "")
    )

    domains_of_focus = parse_csv_field(
        ask("Focus domains (comma-separated, optional): ", "")
    )

    onboarding_survey = validate_survey_preference(
        ask(
            "Onboarding survey preference [examples-first/analogy-first/step-by-step/visual-map] (examples-first): ",
            "examples-first",
        )
    )

    self_explainer_sample = ask(
        "📝 How do you make something click for yourself? In a few sentences, describe what you do when you’re trying to understand a new idea on your own. (optional): ",
        "",
    ).strip()

    calibration_quiz_score = validate_quiz_score(
        ask("Quick quiz score 0-3 (optional): ", "")
    )

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


def show_profile(profile: UserProfile) -> None:
    print("\nCurrent profile:\n")
    print(f"User ID: {profile.user_id}")
    print(f"Display name: {profile.display_name}")
    print(f"Knowledge level: {profile.knowledge_level}")
    print(f"Learning style: {profile.learning_style}")
    print(f"Interests: {', '.join(profile.interests) if profile.interests else 'None'}")
    print(f"Focus domains: {', '.join(profile.domains_of_focus) if profile.domains_of_focus else 'None'}")
    print(f"Onboarding survey: {profile.onboarding_survey or 'None'}")
    print(
        f"Self-explainer sample: {profile.self_explainer_sample or 'None'}"
    )
    quiz_score_text = (
        str(profile.calibration_quiz_score)
        if profile.calibration_quiz_score >= 0
        else "None"
    )
    print(f"Calibration quiz score: {quiz_score_text}\n")


def show_commands() -> None:
    print("\nCommands:")
    print("- /profile: view current profile")
    print("- /update: re-run onboarding and save profile")
    print("- /help: show available commands")
    print("- /quit: exit")


def main() -> None:
    print("Explain-It-Like-I'm-Me (EILIM)")
    print("Adaptive tutor CLI\n")

    storage = JSONStorage(root="data")
    engine = EILIMEngine()
    llm = LLMExplainer()

    user_id = normalize_user_id(ask("Your user ID: ", "guest"))
    profile = storage.load_profile(user_id)

    if profile is None:
        profile = create_profile(user_id)
        storage.save_profile(profile)
        print("\nProfile created and saved.")
    else:
        print(f"\nWelcome back, {profile.display_name}.")

    if llm.enabled:
        print(f"LLM mode enabled ({llm.model})")
    else:
        print("LLM mode disabled (set EILIM_LLM_API_KEY to enable). Using local adaptive engine.")

    show_commands()

    while True:
        try:
            topic = ask("\nWhat do you want explained? ")
        except KeyboardInterrupt:
            print("\nSee you next session.")
            break

        normalized_topic = topic.strip().lower()

        if normalized_topic == "/quit":
            print("See you next session.")
            break
        if normalized_topic == "/profile":
            show_profile(profile)
            continue
        if normalized_topic == "/update":
            profile = create_profile(user_id)
            storage.save_profile(profile)
            print("Profile updated.")
            continue
        if normalized_topic == "/help":
            show_commands()
            continue
        if normalized_topic.startswith("/"):
            print("Unknown command. Type /help for available commands.")
            continue
        if not topic:
            print("Please enter a topic or /quit.")
            continue

        recent_topics = storage.recent_topics(user_id, limit=5)

        # Try the original engine/LLM, then fallback to local generator
        explanation = None
        source = "local-fallback"
        domain = "general"
        try:
            explanation, domain, source = generate_explanation(
                topic=topic,
                profile=profile,
                recent_topics=recent_topics,
                engine=engine,
                llm=llm,
            )
        except Exception as e:
            logger.warning("generate_explanation failed: %s. Using fallback.", e)

        if not explanation:
            explanation = local_fallback_explanation(topic, profile, recent_topics)
            source = "local-fallback"

        print("\n" + explanation)
        print(f"Source: {source}")

        interaction = Interaction(
            user_id=user_id,
            topic=topic,
            explanation=explanation,
            domain=domain,
        )
        storage.save_interaction(interaction)

        rating = validate_rating(
            ask("\nRate this explanation 1-5 (or press enter to skip): ", "")
        )
        if rating >= 1:
            comment = ask("Optional feedback comment: ")
            storage.save_feedback(
                Feedback(user_id=user_id, topic=topic, rating=rating, comment=comment)
            )

            history = storage.recent_feedback(user_id=user_id, limit=5)
            latest = history[-1]
            profile, updates = tune_profile_from_feedback(
                profile=profile,
                latest_feedback=latest,
                recent_feedback=history,
            )
            storage.save_profile(profile)

            print("Feedback saved. Thanks.")
            if updates:
                print("Profile auto-tuned from feedback:")
                for update in updates:
                    print(f"- {update}")


if __name__ == "__main__":
    main()
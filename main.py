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
    VALID_KNOWLEDGE_LEVELS,
    VALID_LEARNING_STYLES,
    VALID_SURVEY_PREFERENCES,
)


def ask(prompt: str, default: str = "") -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        return default
    return value if value else default


def create_profile(user_id: str) -> UserProfile:
    print("\nLet us personalize your tutor profile.")
    display_name_raw = ask("Display name: ", user_id)
    display_name = validate_display_name(display_name_raw, user_id)
    knowledge_level = validate_knowledge_level(
        ask("Knowledge level (beginner/intermediate/advanced) [beginner]: ", "beginner")
    )
    learning_style = validate_learning_style(
        ask("Learning style (step-by-step/visual/story/code) [step-by-step]: ", "step-by-step")
    )
    interests = parse_csv_field(
        ask("Interests (comma-separated, e.g., music,gaming,cooking): ", "")
    )
    domains = parse_csv_field(
        ask("Domains you care about (comma-separated, optional): ", "")
    )
    survey = validate_survey_preference(
        ask("Survey preference (examples-first/analogy-first/step-by-step/visual-map) [examples-first]: ", "examples-first")
    )
    self_sample = ask(
        "How would you explain a concept to yourself? (1 short paragraph, optional): ",
        "",
    )
    quiz_score = validate_quiz_score(
        ask("Quick calibration quiz score 0-3 (optional): ", "")
    )

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


def show_profile(profile: UserProfile) -> None:
    print("\nCurrent profile")
    print(f"- Name: {profile.display_name}")
    print(f"- Knowledge level: {profile.knowledge_level}")
    print(f"- Learning style: {profile.learning_style}")
    print(f"- Interests: {', '.join(profile.interests) if profile.interests else 'none'}")
    print(
        "- Focus domains: "
        + (", ".join(profile.domains_of_focus) if profile.domains_of_focus else "none")
    )
    print(f"- Survey preference: {profile.onboarding_survey or 'none'}")
    print(
        f"- Quiz score: {profile.calibration_quiz_score if profile.calibration_quiz_score >= 0 else 'not set'}"
    )


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
        explanation, domain, source = generate_explanation(
            topic=topic,
            profile=profile,
            recent_topics=recent_topics,
            engine=engine,
            llm=llm,
        )

        if source == "llm":
            print("Source: LLM")
        else:
            print("Source: Local adaptive engine")

        print("\n" + explanation)

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

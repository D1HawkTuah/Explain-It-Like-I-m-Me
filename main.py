from eilim import (
    EILIMEngine,
    JSONStorage,
    LLMExplainer,
    UserProfile,
    generate_explanation,
    tune_profile_from_feedback,
)
from eilim.models import Feedback, Interaction


def ask(prompt: str, default: str = "") -> str:
    value = input(prompt).strip()
    return value if value else default


def parse_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def create_profile(user_id: str) -> UserProfile:
    print("\nLet us personalize your tutor profile.")
    display_name = ask("Display name: ", user_id)
    knowledge_level = ask("Knowledge level (beginner/intermediate/advanced) [beginner]: ", "beginner").lower()
    learning_style = ask(
        "Learning style (step-by-step/visual/story/code) [step-by-step]: ",
        "step-by-step",
    ).lower()
    interests = parse_csv(ask("Interests (comma-separated, e.g., music,gaming,cooking): "))
    domains = parse_csv(ask("Domains you care about (comma-separated, optional): "))
    survey = ask(
        "Survey preference (examples-first/analogy-first/step-by-step/visual-map) [examples-first]: ",
        "examples-first",
    ).lower()
    self_sample = ask(
        "How would you explain a concept to yourself? (1 short paragraph, optional): ",
        "",
    )
    quiz_raw = ask("Quick calibration quiz score 0-3 (optional): ", "")

    quiz_score = -1
    if quiz_raw:
        try:
            quiz_score = max(0, min(3, int(quiz_raw)))
        except ValueError:
            quiz_score = -1

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


def main() -> None:
    print("Explain-It-Like-I'm-Me (EILIM)")
    print("Adaptive tutor CLI\n")

    storage = JSONStorage(root="data")
    engine = EILIMEngine()
    llm = LLMExplainer()

    user_id = ask("Your user ID: ", "guest")
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
        print("LLM mode disabled (set OPENAI_API_KEY to enable). Using local adaptive engine.")

    print("\nCommands: /profile to view, /update to edit, /quit to exit")

    while True:
        topic = ask("\nWhat do you want explained? ")

        if topic.lower() == "/quit":
            print("See you next session.")
            break
        if topic.lower() == "/profile":
            show_profile(profile)
            continue
        if topic.lower() == "/update":
            profile = create_profile(user_id)
            storage.save_profile(profile)
            print("Profile updated.")
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

        rating_raw = ask("\nRate this explanation 1-5 (or press enter to skip): ")
        if rating_raw:
            try:
                rating = max(1, min(5, int(rating_raw)))
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
            except ValueError:
                print("Invalid rating. Skipped feedback save.")


if __name__ == "__main__":
    main()

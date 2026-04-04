import os
from typing import Any, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled in runtime logic
    OpenAI = Any

from .models import UserProfile


class LLMExplainer:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("EILIM_OPENAI_MODEL", "gpt-4.1-mini")
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client: Optional[OpenAI] = None

        if self._api_key and OpenAI is not Any:
            self._client = OpenAI(api_key=self._api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def explain(
        self,
        topic: str,
        profile: UserProfile,
        recent_topics: List[str],
        domain_hint: str,
        semantic_context: Optional[dict[str, object]] = None,
    ) -> str:
        if not self._client:
            raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY.")

        system_prompt = (
            "You are EILIM, an adaptive tutor. Tailor explanations to the user profile. "
            "Use clear language, practical examples, and include a short self-check. "
            "If medical or legal topics appear, include a brief caution that this is educational only."
        )
        user_prompt = self._build_user_prompt(
            topic,
            profile,
            recent_topics,
            domain_hint,
            semantic_context,
        )

        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("Model returned an empty explanation.")
        return text

    @staticmethod
    def _build_user_prompt(
        topic: str,
        profile: UserProfile,
        recent_topics: List[str],
        domain_hint: str,
        semantic_context: Optional[dict[str, object]] = None,
    ) -> str:
        interests = ", ".join(profile.interests) if profile.interests else "none provided"
        focus = ", ".join(profile.domains_of_focus) if profile.domains_of_focus else "none provided"
        recent = ", ".join(recent_topics) if recent_topics else "none yet"
        self_sample = profile.self_explainer_sample.strip() if profile.self_explainer_sample else "none provided"
        survey_pref = profile.onboarding_survey.strip() if profile.onboarding_survey else "none provided"
        quiz_score = (
            str(profile.calibration_quiz_score)
            if profile.calibration_quiz_score >= 0
            else "not provided"
        )
        memory_summary = ""
        if semantic_context:
            memory_summary = str(semantic_context.get("summary", "")).strip()
        memory_text = memory_summary if memory_summary else "none yet"

        return (
            f"Topic: {topic}\n"
            f"Domain hint: {domain_hint}\n"
            "User profile:\n"
            f"- Name: {profile.display_name}\n"
            f"- Knowledge level: {profile.knowledge_level}\n"
            f"- Learning style: {profile.learning_style}\n"
            f"- Interests: {interests}\n"
            f"- Focus domains: {focus}\n"
            f"- Onboarding survey preference: {survey_pref}\n"
            f"- Self explainer sample: {self_sample}\n"
            f"- Calibration quiz score (0-3): {quiz_score}\n"
            f"Recent topics: {recent}\n\n"
            f"Conversation memory summary: {memory_text}\n\n"
            "Output format requirements:\n"
            "1) Quick take (2-3 lines)\n"
            "2) Core explanation tailored to profile\n"
            "3) One analogy tied to user interests\n"
            "4) Try-it-yourself mini check\n"
            "5) Optional format flair based on learning style (visual, story, or code when relevant)"
        )
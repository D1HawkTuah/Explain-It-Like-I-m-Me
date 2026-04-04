from typing import List

from .models import UserProfile


class EILIMEngine:
    def explain(
        self,
        topic: str,
        profile: UserProfile,
        recent_topics: List[str],
        semantic_context: dict[str, object] | None = None,
    ) -> str:
        domain = self.infer_domain(topic)
        analogy = self._pick_analogy(profile.interests, topic)
        depth = self._depth_line(profile.knowledge_level, domain)
        style_block = self._style_block(profile.learning_style, topic)
        continuity = self._continuity_line(recent_topics)
        memory_line = self._memory_line(semantic_context)
        self_voice_line = self._self_voice_line(profile)

        return "\n".join(
            [
                f"Topic: {topic}",
                f"Domain guess: {domain}",
                "",
                "Quick take:",
                self._quick_take(topic, profile.knowledge_level),
                "",
                "Core explanation:",
                depth,
                f"Think of it like this: {analogy}",
                continuity,
                memory_line,
                self_voice_line,
                "",
                "Try it yourself:",
                self._check_yourself(topic, profile.knowledge_level),
                "",
                style_block,
            ]
        ).strip()

    def infer_domain(self, topic: str) -> str:
        return self._infer_domain(topic)

    @staticmethod
    def _infer_domain(topic: str) -> str:
        text = topic.lower()
        if any(word in text for word in ["math", "algebra", "calculus", "equation"]):
            return "school-math"
        if any(word in text for word in ["chem", "atom", "reaction", "molecule"]):
            return "school-chemistry"
        if any(word in text for word in ["history", "war", "empire", "revolution"]):
            return "school-history"
        if any(word in text for word in ["budget", "credit", "loan", "tax", "invest"]):
            return "personal-finance"
        if any(word in text for word in ["medical", "symptom", "disease", "blood", "drug"]):
            return "medical-basics"
        if any(word in text for word in ["wifi", "router", "computer", "phone", "bug"]):
            return "tech-troubleshooting"
        return "general-knowledge"

    @staticmethod
    def _quick_take(topic: str, level: str) -> str:
        if level == "advanced":
            return (
                f"{topic} is easiest to understand as a system with inputs, rules, and outputs. "
                "If you track all three, most confusion disappears."
            )
        if level == "intermediate":
            return (
                f"{topic} is about connecting cause and effect. "
                "If you spot what changes and what stays fixed, it clicks fast."
            )
        return (
            f"{topic} is just a way to answer a practical question. "
            "We break it into tiny pieces and solve one piece at a time."
        )

    @staticmethod
    def _depth_line(level: str, domain: str) -> str:
        if level == "advanced":
            return (
                f"At an advanced level in {domain}, focus on edge-cases, assumptions, and trade-offs. "
                "Ask what happens when conditions are extreme or noisy."
            )
        if level == "intermediate":
            return (
                f"At an intermediate level in {domain}, focus on the mechanism: "
                "what triggers each step and why each step matters."
            )
        return (
            f"At a beginner level in {domain}, focus on vocabulary first, then one concrete example, "
            "then repeat with a second example."
        )

    @staticmethod
    def _pick_analogy(interests: List[str], topic: str) -> str:
        if not interests:
            return f"learning {topic} is like using a map: landmarks first, details second"

        favorite = interests[0].lower()
        if "sports" in favorite:
            return "it is like game strategy: read the field, pick a play, then adjust after each move"
        if "music" in favorite:
            return "it is like learning chords: simple patterns first, then richer combinations"
        if "gaming" in favorite or "games" in favorite:
            return "it is like a game skill tree: unlock core skills before advanced builds"
        if "cooking" in favorite:
            return "it is like cooking: master heat, timing, and ingredients before complex recipes"
        if "cars" in favorite or "mechanic" in favorite:
            return "it is like diagnosing a car: identify the subsystem, test one variable at a time"
        return f"it is like your interest in {favorite}: start with core patterns, then layer details"

    @staticmethod
    def _continuity_line(recent_topics: List[str]) -> str:
        if not recent_topics:
            return "This is our first topic together, so we will calibrate as we go."
        if len(recent_topics) == 1:
            return f"Compared with your previous topic ({recent_topics[-1]}), this follows a similar learn-build-test loop."
        joined = ", ".join(recent_topics[-2:])
        return f"This connects to your recent topics ({joined}) by reusing the same break-it-down approach."

    @staticmethod
    def _memory_line(semantic_context: dict[str, object] | None) -> str:
        if not semantic_context:
            return ""

        summary = str(semantic_context.get("summary", "")).strip()
        if not summary:
            return ""
        return f"Conversation memory: {summary}"

    @staticmethod
    def _self_voice_line(profile: UserProfile) -> str:
        hints: List[str] = []

        if profile.onboarding_survey:
            hints.append(f"survey preference: {profile.onboarding_survey}")
        if profile.self_explainer_sample:
            sample = " ".join(profile.self_explainer_sample.split())
            if len(sample) > 120:
                sample = sample[:117].rstrip() + "..."
            hints.append(f"self-explainer sample: {sample}")
        if profile.calibration_quiz_score >= 0:
            hints.append(f"quiz score: {profile.calibration_quiz_score}/3")

        if not hints:
            return ""
        return "Explain-like-me cues: " + " | ".join(hints)

    @staticmethod
    def _check_yourself(topic: str, level: str) -> str:
        if level == "advanced":
            return (
                f"Explain {topic} in 3 layers: baseline model, failure mode, and mitigation. "
                "If you can do that, your understanding is strong."
            )
        if level == "intermediate":
            return (
                f"Describe {topic} to a friend using one example and one counterexample. "
                "If both make sense, you understand the mechanism."
            )
        return (
            f"Give one real-life example of {topic} and name the first step you would take. "
            "If you can do that, you already have a useful understanding."
        )

    @staticmethod
    def _style_block(learning_style: str, topic: str) -> str:
        style = learning_style.lower()
        if style == "visual":
            return "\n".join(
                [
                    "Visual map:",
                    f"[Question about {topic}] -> [Key idea] -> [Example] -> [Check understanding]",
                ]
            )
        if style == "code":
            return "\n".join(
                [
                    "Code-style summary:",
                    "```python",
                    "def understand(topic):",
                    "    idea = identify_core_idea(topic)",
                    "    example = build_example(idea)",
                    "    return test_understanding(example)",
                    "```",
                ]
            )
        if style == "story":
            return (
                "Story mode: imagine someone who gets stuck at first, finds one clear pattern, "
                "then uses that pattern to solve a harder version."
            )
        return "Step-by-step mode: define terms, walk one example slowly, then solve a similar one together."

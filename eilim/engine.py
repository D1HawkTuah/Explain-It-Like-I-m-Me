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
        topic_label = self._topic_label(topic)
        analogy = self._pick_analogy(profile.interests, topic)
        quick_take = self._quick_take(topic_label, profile.knowledge_level, domain)
        topic_specific = self._topic_specific_explanation(topic, profile.knowledge_level)
        depth = topic_specific or self._domain_explanation(topic_label, profile.knowledge_level, domain)
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
                quick_take,
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
        if any(word in text for word in ["physics", "gravity", "force", "orbit", "motion", "energy"]):
            return "school-physics"
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
    def _quick_take(topic_label: str, level: str, domain: str) -> str:
        if level == "advanced":
            return (
                f"{topic_label.title()} in {domain} is best understood by modeling mechanism, assumptions, "
                "and limits together."
            )
        if level == "intermediate":
            return (
                f"{topic_label.title()} becomes clearer when you track what causes change, "
                "what stays constant, and what outcome you can predict."
            )
        return (
            f"{topic_label.title()} means understanding what it is, how it works, and where you see it in real life."
        )

    @staticmethod
    def _domain_explanation(topic_label: str, level: str, domain: str) -> str:
        if domain == "school-math":
            if level == "advanced":
                return (
                    f"For {topic_label}, represent the problem symbolically, test assumptions, and check edge cases. "
                    "Most mistakes come from hidden constraints or domain restrictions."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, identify variables, write the relationship as an equation, "
                    "then solve while checking units and signs."
                )
            return (
                f"For {topic_label}, start with known numbers, identify the unknown, and apply one rule at a time. "
                "Example: if two quantities change together, write a simple equation to connect them."
            )

        if domain == "school-physics":
            if level == "advanced":
                return (
                    f"Analyze {topic_label} through conservation laws, reference frames, and model validity. "
                    "Use approximations deliberately and state when they break down."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, connect force, motion, and energy: ask what force acts, "
                    "how velocity changes, and where energy transfers."
                )
            return (
                f"For {topic_label}, ask: what is pushing or pulling, what moves, and how fast it changes. "
                "That gives you a simple cause-and-effect picture."
            )

        if domain == "school-chemistry":
            if level == "advanced":
                return (
                    f"Treat {topic_label} with structure-property relationships, equilibrium dynamics, and energetics. "
                    "Predict behavior from bonding and electron distribution."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, track reactants, products, and conditions like temperature or concentration. "
                    "Then use balanced equations to quantify change."
                )
            return (
                f"For {topic_label}, think of atoms rearranging into new combinations. "
                "A chemical equation is a before-and-after map of that rearrangement."
            )

        if domain == "school-history":
            if level == "advanced":
                return (
                    f"Study {topic_label} by comparing sources, bias, and long-term structural causes "
                    "alongside short-term triggers."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, build a timeline of causes, key events, and consequences. "
                    "Then compare perspectives from different groups involved."
                )
            return (
                f"For {topic_label}, ask three questions: what happened, why it happened, and what changed after. "
                "That gives a usable history summary."
            )

        if domain == "personal-finance":
            if level == "advanced":
                return (
                    f"For {topic_label}, model risk, liquidity, taxes, and compounding together. "
                    "Evaluate trade-offs under different scenarios."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, calculate cash flow first, then compare options by total cost, "
                    "risk exposure, and time horizon."
                )
            return (
                f"For {topic_label}, start with income vs expenses, then check fees, interest, and due dates. "
                "Simple tracking prevents most common mistakes."
            )

        if domain == "medical-basics":
            if level == "advanced":
                return (
                    f"For {topic_label}, connect physiology, differential diagnosis logic, and evidence quality. "
                    "Use this as educational context, not personal medical advice."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, map symptoms to body systems, note likely causes, "
                    "and identify red flags that require professional care."
                )
            return (
                f"For {topic_label}, focus on what the condition is, common signs, and when to seek help. "
                "This is educational information, not a diagnosis."
            )

        if domain == "tech-troubleshooting":
            if level == "advanced":
                return (
                    f"Troubleshoot {topic_label} with layered isolation: hardware, network, OS, app, then config drift. "
                    "Use reproducible checks at each layer."
                )
            if level == "intermediate":
                return (
                    f"For {topic_label}, verify power/connection, test network path, inspect settings, and check logs. "
                    "Change one variable at a time to find the fault."
                )
            return (
                f"For {topic_label}, try a simple order: restart, reconnect, update, and retest. "
                "If one step changes behavior, you found a useful clue."
            )

        if level == "advanced":
            return (
                f"For {topic_label}, define the core model, test assumptions, and note where the model stops working."
            )
        if level == "intermediate":
            return (
                f"For {topic_label}, explain the mechanism in order: inputs, process, and outcome. "
                "Then test with one concrete example."
            )
        return (
            f"For {topic_label}, start with a plain definition, then walk one real-world example step by step."
        )

    @staticmethod
    def _topic_label(topic: str) -> str:
        cleaned = " ".join(topic.strip().split())
        lowered = cleaned.lower()
        prefixes = [
            "teach me about ",
            "explain ",
            "what is ",
            "what are ",
            "how does ",
            "how do ",
            "help me understand ",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                trimmed = cleaned[len(prefix):].strip(" ?.!")
                return trimmed or cleaned
        return cleaned.strip(" ?.!")

    @staticmethod
    def _topic_specific_explanation(topic: str, level: str) -> str:
        text = topic.lower()

        if "gravity" in text:
            if level == "advanced":
                return (
                    "Gravity is modeled by Newtonian attraction at many scales "
                    "($F = G*(m1*m2)/r^2$), while general relativity reframes it as spacetime curvature. "
                    "Use Newton's model for most everyday calculations and switch to relativity in strong fields, "
                    "high precision timing, or near-light-speed regimes."
                )
            if level == "intermediate":
                return (
                    "Gravity is the attraction between masses, and the pull weakens with distance by an inverse-square rule. "
                    "That is why planets stay in orbit and why objects near Earth accelerate downward at about 9.8 m/s^2 "
                    "before air resistance changes what we observe."
                )
            return (
                "Gravity is the pull between things that have mass. "
                "It makes dropped objects fall and keeps the Moon and satellites around Earth. "
                "Near Earth's surface, gravity gives objects a downward acceleration of about 9.8 m/s^2."
            )

        return ""

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

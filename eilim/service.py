from typing import Optional, Tuple

from .engine import EILIMEngine
from .models import UserProfile


def generate_explanation(
    topic: str,
    profile: UserProfile,
    recent_topics: list[str],
    engine: EILIMEngine,
    llm: Optional[object] = None,
    semantic_context: Optional[dict[str, object]] = None,
) -> Tuple[str, str, str]:
    domain = engine.infer_domain(topic)

    if llm is not None and getattr(llm, "enabled", False):
        try:
            try:
                explanation = llm.explain(
                    topic=topic,
                    profile=profile,
                    recent_topics=recent_topics,
                    domain_hint=domain,
                    semantic_context=semantic_context,
                )
            except TypeError:
                explanation = llm.explain(
                    topic=topic,
                    profile=profile,
                    recent_topics=recent_topics,
                    domain_hint=domain,
                )
            if explanation and explanation.strip():
                return explanation, domain, "llm"
        except Exception:
            pass

    local_explanation = engine.explain(
        topic=topic,
        profile=profile,
        recent_topics=recent_topics,
        semantic_context=semantic_context,
    )
    return local_explanation, domain, "local"

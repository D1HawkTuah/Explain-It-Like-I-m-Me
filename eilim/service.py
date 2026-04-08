import logging
from typing import Optional, Tuple

from .engine import EILIMEngine
from .models import UserProfile

logger = logging.getLogger(__name__)


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
            except TypeError as e:
                logger.debug("LLM explain() signature mismatch, retrying without semantic_context", exc_info=True)
                explanation = llm.explain(
                    topic=topic,
                    profile=profile,
                    recent_topics=recent_topics,
                    domain_hint=domain,
                )
            if explanation and explanation.strip():
                logger.info(f"LLM explanation generated for topic: {topic[:50]}")
                return explanation, domain, "llm"
            else:
                logger.warning("LLM returned empty or whitespace-only explanation")
        except RuntimeError as e:
            logger.warning(f"LLM RuntimeError (auth/config issue): {str(e)[:100]}")
        except Exception as e:
            logger.error(f"LLM exception ({type(e).__name__}): {str(e)[:100]}", exc_info=True)

    logger.debug(f"Falling back to local engine for topic: {topic[:50]}")
    local_explanation = engine.explain(
        topic=topic,
        profile=profile,
        recent_topics=recent_topics,
        semantic_context=semantic_context,
    )
    return local_explanation, domain, "local"

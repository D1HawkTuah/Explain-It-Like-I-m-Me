import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import Feedback, Interaction, UserProfile

logger = logging.getLogger(__name__)


class JSONStorage:
    def __init__(self, root: str = "data") -> None:
        self.root = Path(root)
        self.users_dir = self.root / "users"
        self.interactions_file = self.root / "interactions.jsonl"
        self.feedback_file = self.root / "feedback.jsonl"
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def _user_file(self, user_id: str) -> Path:
        safe_id = "".join(ch for ch in user_id.strip().lower() if ch.isalnum() or ch in "-_")
        safe_id = safe_id or "user"
        return self.users_dir / f"{safe_id}.json"

    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        path = self._user_file(user_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return UserProfile.from_dict(payload)
        except json.JSONDecodeError as e:
            logger.error(f"Profile file corrupted ({path}): {str(e)[:80]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading profile ({path}): {str(e)[:80]}")
            return None

    def save_profile(self, profile: UserProfile) -> None:
        path = self._user_file(profile.user_id)
        try:
            path.write_text(
                json.dumps(profile.to_dict(), indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            logger.debug(f"Profile saved for user: {profile.user_id}")
        except Exception as e:
            logger.error(f"Failed to save profile ({path}): {str(e)[:80]}")
            raise

    def save_interaction(self, interaction: Interaction) -> None:
        try:
            self._append_jsonl(self.interactions_file, interaction.to_dict())
            logger.debug(f"Interaction saved for user: {interaction.user_id}, topic: {interaction.topic[:30]}")
        except Exception as e:
            logger.error(f"Failed to save interaction: {str(e)[:80]}")
            raise

    def save_feedback(self, feedback: Feedback) -> None:
        try:
            self._append_jsonl(self.feedback_file, feedback.to_dict())
            logger.debug(f"Feedback saved for user: {feedback.user_id}, rating: {feedback.rating}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {str(e)[:80]}")
            raise

    def recent_feedback(self, user_id: str, limit: int = 5) -> List[Feedback]:
        if not self.feedback_file.exists():
            return []

        items: List[Feedback] = []
        try:
            for line_no, line in enumerate(self.feedback_file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record: Dict[str, object] = json.loads(line)
                    if record.get("user_id") != user_id:
                        continue
                    items.append(
                        Feedback(
                            user_id=str(record.get("user_id", "")),
                            topic=str(record.get("topic", "")),
                            rating=int(record.get("rating", 0)),
                            comment=str(record.get("comment", "")),
                            created_at=str(record.get("created_at", "")),
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON in feedback file at line {line_no}, skipping")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid feedback record at line {line_no}: {str(e)[:60]}")
        except Exception as e:
            logger.error(f"Error reading feedback file: {str(e)[:80]}")
            return []

        return items[-limit:]

    def recent_topics(self, user_id: str, limit: int = 5) -> List[str]:
        if not self.interactions_file.exists():
            return []

        topics: List[str] = []
        try:
            for line_no, line in enumerate(self.interactions_file.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record: Dict[str, str] = json.loads(line)
                    if record.get("user_id") == user_id:
                        topic = record.get("topic", "")
                        if topic:
                            topics.append(topic)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON in interactions file at line {line_no}, skipping")
                except Exception as e:
                    logger.warning(f"Error parsing interaction at line {line_no}: {str(e)[:60]}")
        except Exception as e:
            logger.error(f"Error reading interactions file: {str(e)[:80]}")
            return []

        return topics[-limit:]

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .models import Feedback, Interaction, MasteryRecord, UserProfile

logger = logging.getLogger(__name__)


class JSONStorage:
    def __init__(self, root: str = "data") -> None:
        self.root = Path(root)
        self.users_dir = self.root / "users"
        self.interactions_file = self.root / "interactions.jsonl"
        self.feedback_file = self.root / "feedback.jsonl"
        self.db_path = self.root / "eilim.db"
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_database()

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

    def save_mastery_record(self, record: MasteryRecord) -> None:
        with self._sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO mastery_records
                    (user_id, topic, mastery_score, review_count, ease_factor, interval_days, last_reviewed, next_review_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic) DO UPDATE SET
                    mastery_score = excluded.mastery_score,
                    review_count = excluded.review_count,
                    ease_factor = excluded.ease_factor,
                    interval_days = excluded.interval_days,
                    last_reviewed = excluded.last_reviewed,
                    next_review_at = excluded.next_review_at
                """,
                (
                    record.user_id,
                    record.topic,
                    record.mastery_score,
                    record.review_count,
                    record.ease_factor,
                    record.interval_days,
                    record.last_reviewed,
                    record.next_review_at,
                ),
            )

    def load_mastery_record(self, user_id: str, topic: str) -> Optional[MasteryRecord]:
        with self._sqlite_connection() as conn:
            row = conn.execute(
                "SELECT user_id, topic, mastery_score, review_count, ease_factor, interval_days, last_reviewed, next_review_at FROM mastery_records WHERE user_id = ? AND topic = ?",
                (user_id, topic),
            ).fetchone()

        if not row:
            return None

        return MasteryRecord(
            user_id=row["user_id"],
            topic=row["topic"],
            mastery_score=int(row["mastery_score"]),
            review_count=int(row["review_count"]),
            ease_factor=float(row["ease_factor"]),
            interval_days=int(row["interval_days"]),
            last_reviewed=row["last_reviewed"],
            next_review_at=row["next_review_at"],
        )

    def load_mastery_records(self, user_id: str, limit: int = 50) -> List[MasteryRecord]:
        with self._sqlite_connection() as conn:
            rows = conn.execute(
                "SELECT user_id, topic, mastery_score, review_count, ease_factor, interval_days, last_reviewed, next_review_at FROM mastery_records WHERE user_id = ? ORDER BY next_review_at ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        return [
            MasteryRecord(
                user_id=row["user_id"],
                topic=row["topic"],
                mastery_score=int(row["mastery_score"]),
                review_count=int(row["review_count"]),
                ease_factor=float(row["ease_factor"]),
                interval_days=int(row["interval_days"]),
                last_reviewed=row["last_reviewed"],
                next_review_at=row["next_review_at"],
            )
            for row in rows
        ]

    def due_mastery_topics(self, user_id: str, limit: int = 10) -> List[MasteryRecord]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._sqlite_connection() as conn:
            rows = conn.execute(
                "SELECT user_id, topic, mastery_score, review_count, ease_factor, interval_days, last_reviewed, next_review_at FROM mastery_records WHERE user_id = ? AND next_review_at <= ? ORDER BY next_review_at ASC LIMIT ?",
                (user_id, now_iso, limit),
            ).fetchall()

        return [
            MasteryRecord(
                user_id=row["user_id"],
                topic=row["topic"],
                mastery_score=int(row["mastery_score"]),
                review_count=int(row["review_count"]),
                ease_factor=float(row["ease_factor"]),
                interval_days=int(row["interval_days"]),
                last_reviewed=row["last_reviewed"],
                next_review_at=row["next_review_at"],
            )
            for row in rows
        ]

    def update_mastery_for_feedback(self, user_id: str, topic: str, rating: int) -> MasteryRecord:
        record = self.load_mastery_record(user_id, topic)
        if record is None:
            record = MasteryRecord(user_id=user_id, topic=topic)

        updated = self._apply_mastery_feedback(record, rating)
        self.save_mastery_record(updated)
        return updated

    def mastery_overview(self, user_id: str, limit: int = 5) -> Dict[str, object]:
        records = self.load_mastery_records(user_id, limit=limit)
        now = datetime.now(timezone.utc)
        due = [r for r in records if r.next_review_at <= now.isoformat()]
        next_review_days = None
        if records:
            soonest = min(records, key=lambda item: item.next_review_at)
            next_due = datetime.fromisoformat(soonest.next_review_at)
            next_review_days = max(0, (next_due - now).days)

        return {
            "tracked_topics": len(records),
            "due_count": len(due),
            "due_topics": [r.topic for r in due[:limit]],
            "next_due_days": next_review_days,
        }

    @staticmethod
    def _apply_mastery_feedback(record: MasteryRecord, rating: int) -> MasteryRecord:
        quality = max(0, min(5, rating))
        prior_reviews = record.review_count
        scored_quality = quality * 20
        total_score = record.mastery_score * prior_reviews + scored_quality
        record.review_count = prior_reviews + 1
        record.mastery_score = int(total_score / record.review_count)

        if quality < 3:
            record.interval_days = 1
            record.ease_factor = max(1.3, record.ease_factor - 0.2)
        elif prior_reviews == 0:
            record.interval_days = 1
        elif prior_reviews == 1:
            record.interval_days = 3
        else:
            record.interval_days = max(1, min(30, round(record.interval_days * record.ease_factor)))

        ease_delta = 0.1 - (5 - quality) * 0.08
        record.ease_factor = max(1.3, record.ease_factor + ease_delta)

        now = datetime.now(timezone.utc)
        record.last_reviewed = now.isoformat()
        record.next_review_at = (now + timedelta(days=record.interval_days)).isoformat()
        return record

    def save_conversation_turn(
        self,
        user_id: str,
        role: str,
        text: str,
        source: str | None = None,
        created_at: str | None = None,
    ) -> None:
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        with self._sqlite_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(turn_index) FROM conversations WHERE user_id = ?",
                (user_id,),
            )
            result = cursor.fetchone()
            last_index = result[0] if result and result[0] is not None else -1
            next_index = last_index + 1
            conn.execute(
                "INSERT INTO conversations (user_id, turn_index, role, text, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, next_index, role, text, source or "", created_at),
            )

    def conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, str]]:
        if not self.db_path.exists():
            return []

        with self._sqlite_connection() as conn:
            rows = conn.execute(
                "SELECT role, text, source, created_at FROM conversations WHERE user_id = ? ORDER BY turn_index ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "text": row["text"],
                "source": row["source"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def clear_conversation_history(self, user_id: str) -> None:
        with self._sqlite_connection() as conn:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))

    def _sqlite_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_database(self) -> None:
        with self._sqlite_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_id_turn_index ON conversations (user_id, turn_index)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mastery_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mastery_score INTEGER NOT NULL DEFAULT 0,
                    review_count INTEGER NOT NULL DEFAULT 0,
                    ease_factor REAL NOT NULL DEFAULT 2.5,
                    interval_days INTEGER NOT NULL DEFAULT 1,
                    last_reviewed TEXT NOT NULL,
                    next_review_at TEXT NOT NULL,
                    UNIQUE(user_id, topic)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mastery_user_next_review ON mastery_records (user_id, next_review_at)"
            )

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

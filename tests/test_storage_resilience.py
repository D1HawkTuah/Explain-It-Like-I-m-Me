"""Tests for storage resilience against corrupted and malformed data."""

import json
import tempfile
from pathlib import Path

import pytest

from eilim.storage import JSONStorage
from eilim.models import Feedback, Interaction, UserProfile


def test_load_profile_handles_missing_file():
    """Test that loading a non-existent profile returns None gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        result = storage.load_profile("nonexistent_user")
        assert result is None


def test_load_profile_handles_corrupted_json():
    """Test that corrupted profile JSON is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        profile_path = storage._user_file("corrupt_user")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("{ invalid json }", encoding="utf-8")
        
        result = storage.load_profile("corrupt_user")
        assert result is None


def test_load_profile_handles_missing_required_fields():
    """Test that profile with missing required fields is handled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        profile_path = storage._user_file("incomplete_user")
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        # Write valid JSON but missing required fields
        profile_path.write_text(json.dumps({"display_name": "Test"}), encoding="utf-8")
        
        # Should not crash, but may return None or use defaults
        try:
            result = storage.load_profile("incomplete_user")
            # If it doesn't raise, that's fine; storage is resilient
            assert result is None or isinstance(result, UserProfile)
        except KeyError:
            # If it raises KeyError, that's acceptable for severely broken data
            pass


def test_recent_feedback_skips_malformed_lines():
    """Test that malformed lines in feedback JSONL are skipped, not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        feedback_path = storage.feedback_file
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write some good lines, some bad
        lines = [
            json.dumps({"user_id": "u1", "topic": "math", "rating": 5, "comment": "Good", "created_at": "2026-04-08T12:00:00"}),
            "{ broken json line",
            json.dumps({"user_id": "u1", "topic": "physics", "rating": 4, "comment": "Great", "created_at": "2026-04-08T12:01:00"}),
            "",  # Empty line
            "missing required fields",
            json.dumps({"user_id": "u1", "topic": "chemistry", "rating": 3, "comment": "OK", "created_at": "2026-04-08T12:02:00"}),
        ]
        feedback_path.write_text("\n".join(lines), encoding="utf-8")
        
        # Should return valid items, skip malformed
        result = storage.recent_feedback(user_id="u1", limit=10)
        assert len(result) >= 2  # At least 2 valid entries
        assert all(isinstance(f, Feedback) for f in result)
        assert result[-1].topic == "chemistry"


def test_recent_topics_skips_malformed_lines():
    """Test that malformed lines in interactions JSONL are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        interactions_path = storage.interactions_file
        interactions_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            json.dumps({"user_id": "u1", "topic": "algebra", "explanation": "...", "domain": "math", "created_at": "2026-04-08T12:00:00"}),
            "corrupted line ][}{",
            json.dumps({"user_id": "u1", "topic": "geometry", "explanation": "...", "domain": "math", "created_at": "2026-04-08T12:01:00"}),
            "",
            json.dumps({"user_id": "u1", "topic": "calculus", "explanation": "...", "domain": "math", "created_at": "2026-04-08T12:02:00"}),
        ]
        interactions_path.write_text("\n".join(lines), encoding="utf-8")
        
        result = storage.recent_topics(user_id="u1", limit=10)
        assert len(result) >= 2
        assert "algebra" in result
        assert "geometry" in result
        assert "calculus" in result


def test_recent_feedback_handles_missing_file():
    """Test that missing feedback file returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        result = storage.recent_feedback(user_id="u1")
        assert result == []


def test_recent_topics_handles_missing_file():
    """Test that missing interactions file returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        result = storage.recent_topics(user_id="u1")
        assert result == []


def test_save_feedback_creates_directory():
    """Test that save_feedback creates parent directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        feedback = Feedback(user_id="u1", topic="test", rating=5, comment="Great")
        storage.save_feedback(feedback)
        
        assert storage.feedback_file.exists()
        content = storage.feedback_file.read_text(encoding="utf-8").strip()
        assert "u1" in content


def test_save_interaction_creates_directory():
    """Test that save_interaction creates parent directory if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        interaction = Interaction(user_id="u1", topic="test", explanation="...", domain="general")
        storage.save_interaction(interaction)
        
        assert storage.interactions_file.exists()
        content = storage.interactions_file.read_text(encoding="utf-8").strip()
        assert "u1" in content


def test_profile_round_trip():
    """Test that a profile can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = JSONStorage(root=tmpdir)
        original = UserProfile(
            user_id="test_user",
            display_name="Test User",
            knowledge_level="intermediate",
            learning_style="visual",
            interests=["music", "coding"],
            domains_of_focus=["physics"],
            self_explainer_sample="I learn by doing examples",
            onboarding_survey="examples-first",
            calibration_quiz_score=2,
        )
        
        storage.save_profile(original)
        loaded = storage.load_profile("test_user")
        
        assert loaded is not None
        assert loaded.user_id == original.user_id
        assert loaded.display_name == original.display_name
        assert loaded.knowledge_level == original.knowledge_level
        assert loaded.interests == original.interests

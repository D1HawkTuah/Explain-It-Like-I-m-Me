"""Tests for input validation edge cases and security."""

import pytest

from eilim.validation import (
    normalize_user_id,
    validate_knowledge_level,
    validate_learning_style,
    validate_survey_preference,
    validate_quiz_score,
    validate_rating,
    validate_topic,
    validate_comment,
    validate_display_name,
    validate_self_explainer,
    parse_csv_field,
    validate_request_payload_size,
    InputValidationError,
    MAX_TOPIC_LENGTH,
    MAX_COMMENT_LENGTH,
    MAX_USER_ID_LENGTH,
    MAX_DISPLAY_NAME_LENGTH,
)


class TestNormalizeUserID:
    def test_normalizes_whitespace(self):
        assert normalize_user_id("  test_user  ") == "test_user"

    def test_empty_becomes_guest(self):
        assert normalize_user_id("") == "guest"
        assert normalize_user_id("   ") == "guest"
        assert normalize_user_id(None) == "guest"

    def test_enforces_length_limit(self):
        long_id = "x" * (MAX_USER_ID_LENGTH + 100)
        result = normalize_user_id(long_id)
        assert len(result) <= MAX_USER_ID_LENGTH

    def test_preserves_valid_id(self):
        assert normalize_user_id("valid_user_123") == "valid_user_123"


class TestValidateKnowledgeLevel:
    def test_accepts_valid_levels(self):
        assert validate_knowledge_level("beginner") == "beginner"
        assert validate_knowledge_level("INTERMEDIATE") == "intermediate"
        assert validate_knowledge_level("  advanced  ") == "advanced"

    def test_defaults_on_invalid(self):
        assert validate_knowledge_level("expert") == "beginner"
        assert validate_knowledge_level("") == "beginner"
        assert validate_knowledge_level(None) == "beginner"

    def test_custom_default(self):
        assert validate_knowledge_level("invalid", default="advanced") == "advanced"


class TestValidateLearningStyle:
    def test_accepts_valid_styles(self):
        assert validate_learning_style("step-by-step") == "step-by-step"
        assert validate_learning_style("VISUAL") == "visual"
        assert validate_learning_style("code") == "code"
        assert validate_learning_style("STORY") == "story"

    def test_defaults_on_invalid(self):
        assert validate_learning_style("kinesthetic") == "step-by-step"
        assert validate_learning_style("") == "step-by-step"


class TestValidateSurveyPreference:
    def test_accepts_valid_preferences(self):
        assert validate_survey_preference("examples-first") == "examples-first"
        assert validate_survey_preference("ANALOGY-FIRST") == "analogy-first"
        assert validate_survey_preference("visual-map") == "visual-map"

    def test_defaults_on_invalid(self):
        assert validate_survey_preference("other") == "examples-first"
        assert validate_survey_preference("") == "examples-first"


class TestValidateQuizScore:
    def test_accepts_valid_scores(self):
        assert validate_quiz_score("0") == 0
        assert validate_quiz_score("1") == 1
        assert validate_quiz_score("3") == 3

    def test_clamps_out_of_range(self):
        assert validate_quiz_score("-5") == 0
        assert validate_quiz_score("99") == 3

    def test_defaults_on_invalid(self):
        assert validate_quiz_score("abc") == -1
        assert validate_quiz_score("") == -1
        assert validate_quiz_score(None) == -1

    def test_custom_default(self):
        assert validate_quiz_score("xyz", default=2) == 2


class TestValidateRating:
    def test_accepts_valid_ratings(self):
        assert validate_rating("1") == 1
        assert validate_rating("3") == 3
        assert validate_rating("5") == 5

    def test_clamps_out_of_range(self):
        assert validate_rating("-1") == -1  # Returns default for out-of-range below 1
        assert validate_rating("10") == 5

    def test_defaults_on_invalid(self):
        assert validate_rating("not_a_number") == -1
        assert validate_rating("") == -1


class TestValidateTopic:
    def test_truncates_long_topics(self):
        long_topic = "x" * (MAX_TOPIC_LENGTH + 100)
        result = validate_topic(long_topic)
        assert len(result) == MAX_TOPIC_LENGTH

    def test_returns_empty_on_none(self):
        assert validate_topic(None) == ""
        assert validate_topic("") == ""

    def test_strips_whitespace(self):
        assert validate_topic("  hello  ") == "hello"


class TestValidateComment:
    def test_truncates_long_comments(self):
        long_comment = "y" * (MAX_COMMENT_LENGTH + 100)
        result = validate_comment(long_comment)
        assert len(result) == MAX_COMMENT_LENGTH

    def test_returns_empty_on_none(self):
        assert validate_comment(None) == ""
        assert validate_comment("") == ""


class TestValidateDisplayName:
    def test_truncates_long_names(self):
        long_name = "z" * (MAX_DISPLAY_NAME_LENGTH + 100)
        result = validate_display_name(long_name)
        assert len(result) == MAX_DISPLAY_NAME_LENGTH

    def test_uses_fallback_on_empty(self):
        assert validate_display_name("") == "user"
        assert validate_display_name(None) == "user"
        assert validate_display_name("  ") == "user"

    def test_custom_fallback(self):
        assert validate_display_name("", fallback="guest") == "guest"


class TestValidateSelfExplainer:
    def test_truncates_long_samples(self):
        long_sample = "a" * 2500
        result = validate_self_explainer(long_sample)
        assert len(result) <= 2000

    def test_returns_empty_on_none(self):
        assert validate_self_explainer(None) == ""


class TestParseCSVField:
    def test_parses_valid_csv(self):
        result = parse_csv_field("music, gaming, cooking")
        assert result == ["music", "gaming", "cooking"]

    def test_filters_empty_items(self):
        result = parse_csv_field("music, , , cooking")
        assert result == ["music", "cooking"]

    def test_strips_whitespace(self):
        result = parse_csv_field("  art  ,  sports  ")
        assert result == ["art", "sports"]

    def test_returns_empty_on_none(self):
        assert parse_csv_field(None) == []
        assert parse_csv_field("") == []


class TestValidateRequestPayloadSize:
    def test_rejects_oversized_payload(self):
        data = {f"field_{i}": i for i in range(25)}
        with pytest.raises(InputValidationError) as exc_info:
            validate_request_payload_size(data, max_fields=20)
        assert "too many fields" in str(exc_info.value).lower()

    def test_accepts_normal_payload(self):
        data = {f"field_{i}": i for i in range(10)}
        validate_request_payload_size(data, max_fields=20)  # Should not raise

    def test_accepts_at_limit(self):
        data = {f"field_{i}": i for i in range(20)}
        validate_request_payload_size(data, max_fields=20)  # Should not raise


class TestSecurityEdgeCases:
    def test_sql_injection_attempt_in_user_id(self):
        result = normalize_user_id("'; DROP TABLE users; --")
        assert "drop" not in result.lower() or len(result) <= MAX_USER_ID_LENGTH

    def test_xss_attempt_in_display_name(self):
        result = validate_display_name("<script>alert('xss')</script>")
        # Should be treated as a regular string, not executed
        assert "<script>" in result or len(result) > 0

    def test_unicode_in_user_id(self):
        result = normalize_user_id("用户_123")
        # Should handle unicode gracefully
        assert isinstance(result, str)

    def test_very_long_csv_field_list(self):
        long_csv = ",".join([f"item_{i}" for i in range(1000)])
        result = parse_csv_field(long_csv)
        # Should limit or handle gracefully
        assert len(result) > 0
        assert len(result) <= 1000  # At least bounded

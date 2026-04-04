# Explain-It-Like-I'm-Me (EILIM)

EILIM is a personalized AI tutor that explains topics in a way tailored to each person. Instead of giving one-size-fits-all answers, it adapts language, examples, and depth based on the learner profile.

## What It Does

- Personalized explanations based on user profile
- Multi-domain topic support (school + life topics)
- Context memory from previous interactions
- Feedback loop with explanation ratings
- Multi-format output (plain text, code-style blocks, simple diagrams)
- Multi-turn semantic memory in web chat sessions

## Current Implementation

This repository currently includes a Python CLI version that demonstrates:

- Adaptive explanation engine
- Optional live LLM generation via OpenAI API
- Per-user profile storage in JSON
- Interaction history storage in JSONL
- Feedback collection and storage in JSONL
- Session-based multi-turn chat history in web UI
- CI test workflow on push/PR

## Project Structure

```text
.
├── app.py
├── main.py
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── users/
│   ├── interactions.jsonl
│   └── feedback.jsonl
├── eilim/
│   ├── __init__.py
│   ├── engine.py
│   ├── llm.py
│   ├── models.py
│   ├── service.py
│   └── storage.py
├── tests/
│   ├── test_engine_output_shape.py
│   ├── test_service_fallback.py
│   └── test_tuning.py
└── templates/
	├── explanation_prompt.md
	└── index.html
```

## Quick Start

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Run the app:

	```bash
	python main.py
	```

4. Enter your user ID and complete your profile.
5. Ask for any topic (examples: algebra, taxes, Wi-Fi troubleshooting, photosynthesis).

## Run the Web UI

1. Start the Flask app:

	```bash
	python app.py
	```

2. Open in browser:

	```text
	http://localhost:8000
	```

3. Ask questions and submit feedback from the page.
4. Review the Session Chat History panel and Semantic Memory summary to continue a multi-turn conversation with context carry-over.
5. For debugging/demo, open memory inspector page:

	```text
	http://localhost:8000/memory/inspect
	```

6. For raw structured payload JSON:

	```text
	http://localhost:8000/memory/inspect.json
	```

The HTML memory inspector includes client-side filters for:

- role (user or assistant)
- keyword search
- turn range (start/end)

## Enable OpenAI Responses (Optional)

EILIM now supports live model-generated explanations in addition to the local rule-based engine.

1. Set your API key:

	```bash
	export OPENAI_API_KEY="your_api_key_here"
	```

2. (Optional) Set a model:

	```bash
	export EILIM_OPENAI_MODEL="gpt-4.1-mini"
	```

3. Start the app normally:

	```bash
	python main.py
	```

If no API key is set, EILIM automatically falls back to the local adaptive engine.

## Example Commands in App

- `/profile` to view current personalization profile
- `/update` to update profile settings
- `/quit` to exit

## How Personalization Works

EILIM adapts explanations using:

- `knowledge_level`: beginner, intermediate, advanced
- `learning_style`: step-by-step, visual, story, code
- `interests`: custom analogy generation
- `recent_topics`: continuity and reinforcement
- `onboarding_survey`: preferred structure (examples-first, analogy-first, etc.)
- `self_explainer_sample`: user-written paragraph in their own words
- `calibration_quiz_score`: optional baseline depth signal (0-3)

## Explain-Like-Me Calibration

EILIM supports three practical signals to learn how users explain things to themselves:

- Survey: choose preferred structure (examples-first, analogy-first, etc.)
- Self-explainer paragraph: write a short paragraph in personal wording
- Quick quiz score (0-3): optional baseline for depth

These signals are stored in profile memory and used during explanation generation.

## Data Notes

- User profiles are stored in `data/users/*.json`.
- Session interactions are appended to `data/interactions.jsonl`.
- Ratings/comments are appended to `data/feedback.jsonl`.

## Feedback-Driven Auto-Tuning

EILIM now adjusts profile settings automatically from ratings/comments:

- Low ratings can reduce explanation depth (e.g., advanced -> intermediate).
- Strong ratings over recent interactions can increase depth.
- Comment keywords can switch style (e.g., "visual", "story", "code", "step-by-step").

This tuning updates the stored profile, so future explanations adapt automatically.

Safeguards included:

- Knowledge level only shifts after at least 3 feedback entries.
- Style updates can still occur from explicit comment intent (e.g., "visual").

## Run Tests

```bash
pytest -q
```

Current tests cover:

- LLM -> local fallback behavior
- Explanation output shape consistency
- Feedback-driven profile tuning
- Semantic memory extraction and injection

## CI

GitHub Actions workflow runs tests automatically on:

- Push to `main`
- Pull requests targeting `main`

## Next Iteration Ideas

- Integrate live LLM generation via OpenAI API
- Add a lightweight web UI
- Add topic mastery tracking over time
- Add quiz mode and spaced repetition
- Add exportable learning summaries
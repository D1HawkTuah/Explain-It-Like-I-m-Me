from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# Load user data
try:
    with open("user_data.json", "r") as f:
        user_data = json.load(f)
except:
    user_data = []

def analyze_user_preferences():
    if not user_data:
        return "No prior data."

    styles = {}
    levels = {}

    for entry in user_data:
        styles[entry["style"]] = styles.get(entry["style"], 0) + 1
        levels[entry["level"]] = levels.get(entry["level"], 0) + 1

    favorite_style = max(styles, key=styles.get)
    favorite_level = max(levels, key=levels.get)

    return f"User prefers {favorite_style} explanations at a {favorite_level} level."

def generate_explanation(question, level, style):
    memory = analyze_user_preferences()

    prompt = f"""
    You are a personalized tutor.

    User memory:
    {memory}

    Task:
    Explain the following question.

    Question: {question}

    Requirements:
    - Level: {level}
    - Style: {style}
    - Be clear and adaptive
    - If analogy style, use a strong real-world comparison
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/explain", methods=["POST"])
def explain():
    data = request.json

    question = data["question"]
    level = data["level"]
    style = data["style"]

    explanation = generate_explanation(question, level, style)

    entry = {
        "question": question,
        "level": level,
        "style": style,
        "timestamp": str(datetime.now())
    }

    user_data.append(entry)

    with open("user_data.json", "w") as f:
        json.dump(user_data, f, indent=2)

    return jsonify({"explanation": explanation})

@app.route("/stats")
def stats():
    topic_count = len(user_data)

    styles = {}
    for entry in user_data:
        styles[entry["style"]] = styles.get(entry["style"], 0) + 1

    return jsonify({
        "total_questions": topic_count,
        "styles": styles
    })

if __name__ == "__main__":
    app.run(debug=True)

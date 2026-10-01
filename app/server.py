"""
server.py
----------
Small Flask web server that puts a chat UI in front of the ML chatbot.

Flow:
  Browser (templates/index.html)
      --> POST /api/chat  {session_id, message}
      --> Flask loads/creates a Session for that session_id
      --> chatbot.handle_message() runs the ML intent model + dialogue manager
      --> Flask returns {reply, session_id}

Run:
    python app/server.py
Then open:
    http://127.0.0.1:5000
"""

import os
import sys
import uuid

from flask import Flask, request, jsonify, render_template
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.pkl")

sys.path.append(SRC_DIR)
from chatbot import Session, handle_message  # noqa: E402

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"Model not found at {MODEL_PATH}.\nRun 'python src/train_model.py' first."
    )

model = joblib.load(MODEL_PATH)

# In-memory session store: session_id -> Session object.
# Fine for a local demo; use a real DB/session store in production.
sessions = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id") or str(uuid.uuid4())
    message = (data.get("message") or "").strip()

    if session_id not in sessions:
        sessions[session_id] = Session()

    if not message:
        return jsonify({"reply": "", "session_id": session_id})

    reply = handle_message(sessions[session_id], message, model)
    return jsonify({"reply": reply, "session_id": session_id})


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

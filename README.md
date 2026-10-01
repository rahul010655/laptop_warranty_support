# Laptop Warranty Support Chatbot (ML-based Intent Classification)

A rule-based chatbot answers keywords. This version replaces that with a
**trained machine learning model** that classifies what the user actually
wants (their *intent*), then runs a small dialogue manager that asks for
missing details one question at a time until it can give a final answer.

## How to explain this project (interview pitch)

> "I built a laptop warranty support chatbot that uses NLP and machine
> learning to understand customer intent instead of relying only on
> keyword matching. I converted customer messages into TF-IDF feature
> vectors and trained a Logistic Regression classifier to sort each
> message into one of six intents: greeting, warranty status check,
> damage report, repair request, coverage question, or out-of-scope.
> Once the intent is known, a dialogue manager fills in the missing
> slots — laptop brand, model, serial number, purchase date, damage
> type — asking the user one question at a time, similar to how a real
> support agent would, and gives a brand-specific final answer."

That one paragraph covers the two things interviewers usually probe:
**the ML part** (text vectorization + classification) and **the
engineering part** (state management / slot filling in a conversation).

## Project structure

```
laptop-warranty-chatbot-ml/
├── README.md                  <- you are here
├── requirements.txt            <- Python dependencies
├── data/
│   └── intents.csv             <- labeled training examples (text, intent)
├── src/
│   ├── train_model.py           <- trains & saves the ML model
│   ├── chatbot.py                <- core chatbot logic (intent + dialogue manager)
│   └── utils.py                   <- brand lookup, date/damage-type regex helpers
├── models/
│   └── intent_classifier.pkl      <- trained model (created by train_model.py)
└── app/
    ├── server.py                  <- Flask web server, exposes /api/chat
    └── templates/
        └── index.html              <- chat UI the browser loads
```

## How it works (architecture)

```
User message
     │
     ▼
┌─────────────────────────┐
│ TF-IDF Vectorizer        │   converts text -> numeric feature vector
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Logistic Regression      │   predicts one of 6 intents
│ (the "ML model")         │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Dialogue manager          │   if a slot (brand/model/serial/date/...)
│ (slot filling, in Python) │   is missing, ask for it; otherwise answer
└─────────────────────────┘
     │
     ▼
Final response to user
```

- **Intents** the model recognizes: `greeting`, `warranty_status`,
  `damage_report`, `repair_request`, `coverage_info`, `out_of_scope`.
- **Slots** the dialogue manager tracks: `brand`, `model`, `serial`,
  `purchase_date`, `damage_type`, `damage_date`, `adp` (Accidental
  Damage Protection yes/no).
- A small safety rule: if a known laptop brand is mentioned, the bot
  never refuses the message as "out of scope," even if the ML model's
  confidence on a short message like "Dell" is low — this compensates
  for the training set being intentionally small.

## Running it (web chat UI)

```bash
cd laptop-warranty-chatbot-ml
pip install -r requirements.txt

# 1. Train the model (reads data/intents.csv, writes models/intent_classifier.pkl)
python src/train_model.py

# 2. Start the web server
python app/server.py
```

Then open **http://127.0.0.1:5000** in your browser — you'll get the same
clean chat interface as before, except every message now goes through the
trained ML model on the backend instead of hand-written regex rules.

```
Browser (index.html)  --fetch POST /api/chat-->  Flask server.py
                                                        │
                                                        ▼
                                          chatbot.py: ML intent model
                                          + slot-filling dialogue manager
```

### Running it in the terminal instead (optional)

```bash
python src/chatbot.py
```

Example session:

```
Bot: Hello! Tell me your laptop brand and what's going on.
You: dell
Bot: Thanks! I see you have a Dell laptop. What would you like help with —
     checking warranty status, reporting damage, or requesting a repair?
You: warranty check
Bot: Got it, a Dell. What's the model name or number?
You: Inspiron 15
Bot: Thanks. Could you share the Service Tag for this Dell?
You: SVC12345
Bot: Last one — what was the purchase date? (dd/mm/yyyy)
You: 12/05/2024
Bot: Here's what I have for your Dell laptop: ...
```

## Honest limitations (good to mention if asked)

- The training set (`data/intents.csv`) is small (~60 examples across 6
  classes), built for demonstration. Accuracy on held-out test rows is
  moderate — in a real deployment you'd collect hundreds/thousands of
  real customer messages per intent and retrain.
- Slot extraction (brand, date, damage type) uses regex, not a trained
  NER model — a natural next step would be to train or fine-tune a
  named-entity recognition model for that instead.
- There's no live warranty database connected; responses point the
  user to the manufacturer's real support site to confirm the actual
  expiration date.

## Possible extensions to mention/build further

- Swap Logistic Regression for a small neural network (e.g. an
  embedding + dense layers, or fine-tune a transformer like DistilBERT)
  once more labeled data is available.
- Add a confidence threshold: if the model isn't confident, ask the
  user to clarify instead of guessing.
- Log real conversations (with consent) to grow the training set over
  time — a feedback loop from production data back into retraining.
- Swap the in-memory session store in `server.py` for Redis or a
  database so conversations survive a server restart.

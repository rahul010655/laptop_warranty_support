"""
chatbot.py
-----------
Console chatbot for Laptop Warranty Support.

Architecture:
  1. ML layer (intent classification):
     The trained TF-IDF + Logistic Regression model (models/intent_classifier.pkl)
     reads the user's raw message and predicts one of:
       greeting | warranty_status | damage_report | repair_request |
       coverage_info | out_of_scope

  2. Dialogue management layer (slot filling):
     Once the intent is known, the bot asks one question at a time for
     whatever details are still missing (brand, model, serial number,
     purchase date, damage type, damage date, accidental damage protection),
     using simple regex helpers in utils.py to parse free-text answers.

  3. Response layer:
     When all required slots for the current intent are filled, the bot
     gives a final, brand-specific answer.

Run:
    python src/chatbot.py
(Run train_model.py first if models/intent_classifier.pkl doesn't exist yet.)
"""

import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import detect_brand, detect_date, detect_damage_type, detect_yes_no, brand_info

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier.pkl")

REFUSAL = ("I am a Laptop Warranty Support Assistant. I can only assist with "
           "laptop warranty, repair, service, and support-related inquiries.")


class Session:
    def __init__(self):
        self.flow = None       # 'warranty' | 'damage' | 'repair' | None
        self.pending = None    # which slot we're waiting on next
        self.brand = None
        self.model = None
        self.serial = None
        self.purchase_date = None
        self.damage_type = None
        self.damage_date = None
        self.adp = None


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run: python src/train_model.py")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predict_intent(model, text):
    return model.predict([text])[0]


def continue_flow(s: Session):
    info = brand_info(s.brand)

    if s.flow in ("warranty", "repair"):
        if not s.brand:
            s.pending = "brand"
            return "Sure — which laptop brand is this? (e.g. Dell, HP, Lenovo, Asus, Acer, Apple, MSI, Samsung)"
        if not s.model:
            s.pending = "model"
            return f"Got it, a {info['name']}. What's the model name or number (e.g. Inspiron 15, Pavilion 14)?"
        if not s.serial:
            s.pending = "serial"
            return f"Thanks. Could you share the {info['id_label']} for this {info['name']}?"
        if not s.purchase_date:
            s.pending = "purchase_date"
            return "Last one — what was the purchase date? (dd/mm/yyyy)"

        s.pending = None
        if s.flow == "warranty":
            s.flow = None
            return (f"Here's what I have for your {info['name']} laptop:\n"
                     f"  - Model: {s.model}\n"
                     f"  - {info['id_label']}: {s.serial}\n"
                     f"  - Purchase date: {s.purchase_date}\n"
                     f"I don't have live access to the warranty database here, so I can't confirm the exact "
                     f"expiration. The fastest way to check is through {info['site']}.")
        else:
            s.flow = None
            return (f"Here's how to proceed with the repair for your {info['name']} laptop:\n"
                     f"  - Documents needed: proof of purchase (invoice), warranty card if issued, valid ID\n"
                     f"  - On file: Model {s.model}, {info['id_label']} {s.serial}, purchased {s.purchase_date}\n"
                     f"  - Next step: raise a repair ticket through {info['site']}, which will also confirm "
                     f"whether it's covered or chargeable")

    if s.flow == "damage":
        if not s.damage_type:
            s.pending = "damage_type"
            who = f" about your {info['name']}" if info else ""
            return f"Sorry to hear that{who}. What kind of damage is it — a screen crack, liquid spill, physical drop, or something else?"
        if not s.damage_date:
            s.pending = "damage_date"
            return "When did this happen?"
        if s.adp is None:
            s.pending = "adp"
            return "Do you have Accidental Damage Protection (ADP) on this laptop? (yes/no)"

        s.pending = None
        s.flow = None
        if s.adp:
            return (f"Thanks for confirming. Since this laptop has Accidental Damage Protection, "
                     f"{s.damage_type} from {s.damage_date} is typically covered, subject to inspection. "
                     f"Please raise a repair ticket" +
                     (f" through {info['site']}" if info else " with your brand's service center") +
                     " and bring proof of purchase plus the ADP certificate.")
        else:
            return (f"Without Accidental Damage Protection, {s.damage_type} usually isn't covered under the "
                     f"standard manufacturer warranty, which covers manufacturing defects only — not accidental "
                     f"or physical damage. The service center" +
                     (f" ({info['site']})" if info else "") +
                     " can give you a repair cost estimate.")

    # No flow yet but brand known -> ask what they need
    if s.brand:
        s.pending = "flow_choice"
        return f"Thanks! I see you have a {info['name']} laptop. What would you like help with — checking warranty status, reporting damage, or requesting a repair?"

    s.pending = None
    return "I can help once I know a bit more. Could you tell me your laptop brand and what issue you're facing — warranty status, damage, or a repair?"


def handle_pending(s: Session, text, model):
    if s.pending == "flow_choice":
        intent = predict_intent(model, text)
        if intent in ("warranty_status", "damage_report", "repair_request"):
            s.flow = {"warranty_status": "warranty", "damage_report": "damage", "repair_request": "repair"}[intent]
            return continue_flow(s)
        return "Just to confirm — is this about warranty status, reporting damage, or a repair request?"

    if s.pending == "brand":
        s.brand = detect_brand(text) or text.strip()
        return continue_flow(s)
    if s.pending == "model":
        s.model = text.strip()
        return continue_flow(s)
    if s.pending == "serial":
        s.serial = text.strip()
        return continue_flow(s)
    if s.pending == "purchase_date":
        s.purchase_date = detect_date(text) or text.strip()
        return continue_flow(s)
    if s.pending == "damage_type":
        s.damage_type = detect_damage_type(text) or text.strip()
        return continue_flow(s)
    if s.pending == "damage_date":
        s.damage_date = text.strip()
        return continue_flow(s)
    if s.pending == "adp":
        yn = detect_yes_no(text)
        s.adp = False if yn is None else yn
        return continue_flow(s)

    return continue_flow(s)


def handle_message(s: Session, text, model):
    if s.pending:
        return handle_pending(s, text, model)

    intent = predict_intent(model, text)
    b = detect_brand(text)
    if b and not s.brand:
        s.brand = b

    # Safety net: with a small training set, a bare brand name (e.g. "Dell")
    # can be misclassified as out_of_scope. A recognized brand mention should
    # never be refused — fall through to the brand-only flow instead.
    if intent == "out_of_scope" and not b:
        return REFUSAL
    if intent == "greeting":
        return "Hello, welcome to Laptop Warranty Support. Tell me your laptop brand and what's going on, and I'll take it from there."
    if intent == "coverage_info":
        info = brand_info(s.brand)
        extra = f" for your {info['name']}" if info else ""
        return (f"Warranty coverage{extra} typically includes manufacturing defects and hardware failure under "
                f"normal use, and excludes accidental damage, liquid damage, and unauthorized repairs — unless "
                f"Accidental Damage Protection was purchased separately.")

    if intent == "warranty_status":
        s.flow = "warranty"
        return continue_flow(s)
    if intent == "damage_report":
        s.flow = "damage"
        dt = detect_damage_type(text)
        if dt:
            s.damage_type = dt
        return continue_flow(s)
    if intent == "repair_request":
        s.flow = "repair"
        return continue_flow(s)

    if b:
        return continue_flow(s)

    return "I can help with warranty status, damage, or repair requests for your laptop. Could you tell me the laptop brand to start?"


def main():
    model = load_model()
    s = Session()
    print("Laptop Warranty Support Assistant (type 'quit' to exit)\n")
    print("Bot: Hello! Tell me your laptop brand and what's going on.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        reply = handle_message(s, text, model)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()

from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import logging

app = Flask(__name__)

# ---------- Load Model Files ----------
MODEL_PATH = "models/enhanced_dtc.pkl"
SCALER_PATH = "models/enhanced_scaler.pkl"
ENCODER_PATH = "models/enhanced_label_encoder.pkl"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to load model artifacts; if they are missing or invalid, keep None
model = None
scaler = None
le = None

def try_load(path):
    if not os.path.exists(path):
        logger.warning("Model file not found: %s", path)
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.exception("Failed to load %s: %s", path, e)
        return None

model = try_load(MODEL_PATH)
scaler = try_load(SCALER_PATH)
le = try_load(ENCODER_PATH)

# ---------- Feature Definitions ----------
SYMPTOMS = [
    'fever', 'coughing', 'vomiting', 'lethargy', 'loss_of_appetite',
    'diarrhea', 'breathing_difficulty', 'weight_loss', 'excessive_drinking',
    'excessive_urination', 'limping', 'skin_problems', 'eye_discharge',
    'nasal_discharge', 'aggression'
]

VALID_ANIMALS = ['Dog', 'Cat', 'Bird', 'Rabbit', 'Hamster', 'Guinea Pig', 'Ferret']

# --- Symptom groups and severity scoring (must match training)
SYMPTOM_GROUPS = {
    'respiratory': {
        'symptoms': ['coughing', 'breathing_difficulty', 'nasal_discharge'],
        'severity_weight': 2.0
    },
    'digestive': {
        'symptoms': ['vomiting', 'diarrhea', 'loss_of_appetite'],
        'severity_weight': 1.5
    },
    'systemic': {
        'symptoms': ['fever', 'lethargy', 'weight_loss'],
        'severity_weight': 1.8
    },
    'urinary': {
        'symptoms': ['excessive_drinking', 'excessive_urination'],
        'severity_weight': 1.3
    },
    'external': {
        'symptoms': ['skin_problems', 'eye_discharge', 'limping', 'aggression'],
        'severity_weight': 1.0
    }
}

def calculate_severity_score(values):
    score = 0
    for group in SYMPTOM_GROUPS.values():
        total = sum(values.get(s, 0) for s in group['symptoms'])
        if total > 0:
            score += (total / len(group['symptoms'])) * group['severity_weight']
    return score


@app.route("/")
def home():
    # indicate to the UI whether the model is available
    model_ready = model is not None and scaler is not None and le is not None
    return render_template("index.html", animals=VALID_ANIMALS, symptoms=SYMPTOMS, model_ready=model_ready)


@app.route("/submit", methods=["POST"])
def submit():
    try:
        # 1. animal
        animal_name = request.form.get("animalName")
        if animal_name not in VALID_ANIMALS:
            return jsonify({"error": "Invalid animal"}), 400

        # 2. symptoms vector
        symptom_values = []
        for sym in SYMPTOMS:
            val = request.form.get(f"symptom_{sym}")
            symptom_values.append(1 if val == "1" else 0)

        # If no symptoms selected, treat as healthy
        if sum(symptom_values) == 0:
            result = {
                "status": "Animal is healthy",
                "confidence": "100.0%"
            }
            return render_template("output.html", result=result)

        # compute severity score to match training feature
        symptom_map = {s: symptom_values[idx] for idx, s in enumerate(SYMPTOMS)}
        severity = calculate_severity_score(symptom_map)

        # 3. encode animal
        if le is None:
            return jsonify({"error": "Model artifacts not loaded. Run training script to generate models."}), 500

        animal_encoded = le.transform([animal_name])[0]

        # 4. build final feature vector (animal, symptoms..., severity_score)
        features = np.array([animal_encoded] + symptom_values + [severity]).reshape(1, -1)

        # 5. scale symptoms + severity (all columns after animal index)
        if scaler is None:
            return jsonify({"error": "Scaler not loaded. Run training script to generate models."}), 500

        scaled = scaler.transform(features[:, 1:])
        features[:, 1:] = scaled

        # 6. prediction
        if model is None:
            return jsonify({"error": "Model not loaded. Run training script to generate models."}), 500

        pred = model.predict(features)[0]
        # some models may not implement predict_proba
        try:
            prob = model.predict_proba(features)[0]
        except Exception:
            prob = [0.0, 1.0] if pred == 1 else [1.0, 0.0]

        result = {
            "status": "Animal is healthy" if pred == 0 else "Animal may have disease",
            "confidence": f"{max(prob) * 100:.1f}%"
        }

        return render_template("output.html", result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os

animals = ['Dog', 'Cat', 'Bird', 'Rabbit', 'Hamster', 'Guinea Pig', 'Ferret']

symptoms = [
    'fever', 'coughing', 'vomiting', 'lethargy', 'loss_of_appetite',
    'diarrhea', 'breathing_difficulty', 'weight_loss', 'excessive_drinking',
    'excessive_urination', 'limping', 'skin_problems', 'eye_discharge',
    'nasal_discharge', 'aggression'
]

# --- Symptom Groups ---
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
        total = sum(values[s] for s in group['symptoms'])
        if total > 0:
            score += (total / len(group['symptoms'])) * group['severity_weight']
    return score


def generate_synthetic_data(n=3000):
    data = []
    for _ in range(n):
        animal = np.random.choice(animals)
        vals = {sym: 0 for sym in symptoms}

        # random noise
        for sym in symptoms:
            if np.random.random() < 0.05:
                vals[sym] = 1

        is_sick = np.random.random() < 0.25
        if is_sick:
            num_groups = np.random.randint(1, 4)
            chosen = np.random.choice(list(SYMPTOM_GROUPS.keys()), num_groups, replace=False)
            for grp in chosen:
                for sym in SYMPTOM_GROUPS[grp]['symptoms']:
                    if np.random.random() < 0.7:
                        vals[sym] = 1

        score = calculate_severity_score(vals)

        if score > 2.5:
            disease = 1
        elif score < 1.0:
            disease = 0
        else:
            disease = 1 if np.random.random() < (score - 1) / 1.5 else 0

        # include severity score as an explicit feature
        row = [animal] + list(vals.values()) + [score] + [disease]
        data.append(row)

    cols = ['animal_name'] + symptoms + ['severity_score'] + ['disease']
    return pd.DataFrame(data, columns=cols)


# Generate
df = generate_synthetic_data(3000)

X = df[['animal_name'] + symptoms + ['severity_score']]
y = df['disease']

le = LabelEncoder()
X['animal_name'] = le.fit_transform(X['animal_name'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
scale_cols = symptoms + ['severity_score']
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=6,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_bal, y_train_bal)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/enhanced_dtc.pkl", "wb"))
pickle.dump(scaler, open("models/enhanced_scaler.pkl", "wb"))
pickle.dump(le, open("models/enhanced_label_encoder.pkl", "wb"))

print("Model + Scaler + Encoder saved successfully.")

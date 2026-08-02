"""
Telecom Churn Prediction — Flask Application
"""

import os, pickle, json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_DATA = None

def load_model():
    global MODEL_DATA
    path = "models/churn_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            MODEL_DATA = pickle.load(f)
        print(f"✓ Model loaded  ({MODEL_DATA['best_name']})")
    else:
        print("⚠ models/churn_model.pkl not found — run train_model.py first")

load_model()

# ── Encoding maps (mirror train_model.py) ────────────────────────────────────
ENC = {
    "gender":           {"Male":1, "Female":0},
    "Partner":          {"Yes":1, "No":0},
    "Dependents":       {"Yes":1, "No":0},
    "PhoneService":     {"Yes":1, "No":0},
    "MultipleLines":    {"Yes":2, "No":1, "No phone service":0},
    "InternetService":  {"DSL":0, "Fiber optic":1, "No":2},
    "OnlineSecurity":   {"Yes":2, "No":1, "No internet service":0},
    "OnlineBackup":     {"Yes":2, "No":1, "No internet service":0},
    "DeviceProtection": {"Yes":2, "No":1, "No internet service":0},
    "TechSupport":      {"Yes":2, "No":1, "No internet service":0},
    "StreamingTV":      {"Yes":2, "No":1, "No internet service":0},
    "StreamingMovies":  {"Yes":2, "No":1, "No internet service":0},
    "Contract":         {"Month-to-month":0, "One year":1, "Two year":2},
    "PaperlessBilling": {"Yes":1, "No":0},
    "PaymentMethod":    {"Bank transfer (automatic)":0, "Credit card (automatic)":1,
                         "Electronic check":2, "Mailed check":3},
}

def preprocess(form):
    tenure  = float(form.get("tenure", 12))
    monthly = float(form.get("MonthlyCharges", 65))
    total   = float(form.get("TotalCharges", tenure * monthly))
    internet = form.get("InternetService", "DSL")
    phone    = form.get("PhoneService", "Yes")

    cpm = total / (tenure + 1)
    tg  = 0 if tenure<=12 else 1 if tenure<=24 else 2 if tenure<=48 else 3
    ns  = sum([
        phone == "Yes",
        internet != "No",
        form.get("OnlineSecurity")  == "Yes",
        form.get("OnlineBackup")    == "Yes",
        form.get("DeviceProtection")== "Yes",
        form.get("TechSupport")     == "Yes",
        form.get("StreamingTV")     == "Yes",
        form.get("StreamingMovies") == "Yes",
    ])

    raw = {
        "gender":           ENC["gender"].get(form.get("gender","Male"), 1),
        "SeniorCitizen":    int(form.get("SeniorCitizen", 0)),
        "Partner":          ENC["Partner"].get(form.get("Partner","No"), 0),
        "Dependents":       ENC["Dependents"].get(form.get("Dependents","No"), 0),
        "tenure":           tenure,
        "PhoneService":     ENC["PhoneService"].get(phone, 1),
        "MultipleLines":    ENC["MultipleLines"].get(form.get("MultipleLines","No"), 1),
        "InternetService":  ENC["InternetService"].get(internet, 0),
        "OnlineSecurity":   ENC["OnlineSecurity"].get(form.get("OnlineSecurity","No internet service"), 0),
        "OnlineBackup":     ENC["OnlineBackup"].get(form.get("OnlineBackup","No internet service"), 0),
        "DeviceProtection": ENC["DeviceProtection"].get(form.get("DeviceProtection","No internet service"), 0),
        "TechSupport":      ENC["TechSupport"].get(form.get("TechSupport","No internet service"), 0),
        "StreamingTV":      ENC["StreamingTV"].get(form.get("StreamingTV","No internet service"), 0),
        "StreamingMovies":  ENC["StreamingMovies"].get(form.get("StreamingMovies","No internet service"), 0),
        "Contract":         ENC["Contract"].get(form.get("Contract","Month-to-month"), 0),
        "PaperlessBilling": ENC["PaperlessBilling"].get(form.get("PaperlessBilling","Yes"), 1),
        "PaymentMethod":    ENC["PaymentMethod"].get(form.get("PaymentMethod","Electronic check"), 2),
        "MonthlyCharges":   monthly,
        "TotalCharges":     total,
        "ChargesPerMonth":  cpm,
        "TenureGroup":      tg,
        "NumServices":      ns,
    }
    feat_order = MODEL_DATA["feature_names"]
    return np.array([[raw.get(f, 0) for f in feat_order]])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           results=MODEL_DATA["results"] if MODEL_DATA else {},
                           model_loaded=MODEL_DATA is not None)

@app.route("/predict", methods=["POST"]) 
def predict():
    if not MODEL_DATA:
        return jsonify({"error": "Model not loaded"}), 500
    form = request.form.to_dict()
    X    = preprocess(form)
    model = MODEL_DATA["model"]
    prob  = float(model.predict_proba(X)[0][1])
    pct   = round(prob * 100, 1)

    if prob >= 0.65:
        level = "High Risk"; color = "#EF4444"
        rec   = ("🚨 Immediate retention action required. Offer a personalised package: "
                 "discounted plan, loyalty bonus, or dedicated account manager.")
    elif prob >= 0.35:
        level = "Medium Risk"; color = "#F59E0B"
        rec   = ("⚠️ Proactive outreach recommended. Send a satisfaction survey and "
                 "offer a flexible billing option or service upgrade.")
    else:
        level = "Low Risk"; color = "#10B981"
        rec   = ("✅ Customer appears loyal. Maintain service quality and consider "
                 "a loyalty or referral rewards programme.")

    return jsonify({
        "probability": pct,
        "prediction":  int(prob >= 0.5),
        "churn_label": "Will Churn" if prob >= 0.5 else "Will Retain",
        "risk_level":  level,
        "risk_color":  color,
        "recommendation": rec,
        "confidence":  round((pct if prob>=0.5 else 100-pct), 1),
    })

@app.route("/analytics")
def analytics():
    return render_template("analytics.html",
                           results=MODEL_DATA["results"] if MODEL_DATA else {},
                           best=MODEL_DATA["best_name"] if MODEL_DATA else "—",
                           n_samples=MODEL_DATA["n_samples"] if MODEL_DATA else 0,
                           churn_rate=round(MODEL_DATA["churn_rate"]*100,1) if MODEL_DATA else 0)

@app.route("/api/results")
def api_results():
    if not MODEL_DATA:
        return jsonify({"error": "Model not loaded"})
    return jsonify(MODEL_DATA["results"])

@app.route("/api/feature_importance")
def api_fi():
    if not MODEL_DATA:
        return jsonify({"error": "Model not loaded"})
    rf = MODEL_DATA["all_models"]["Random Forest"]
    fi = dict(zip(MODEL_DATA["feature_names"], rf.feature_importances_.tolist()))
    return jsonify(dict(sorted(fi.items(), key=lambda x: -x[1])[:15]))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

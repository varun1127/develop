# 📡 TelePredict — Telecom Customer Churn Prediction

A full-stack ML web application predicting customer churn probability using 5 machine learning algorithms.

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates dataset + charts + model.pkl)
python train_model.py

# 4. Run the Flask app
python app.py
# → Open http://localhost:5000
```

## 📁 Project Structure

```
telecom_project/
├── app.py               # Flask application (routes + inference)
├── train_model.py       # ML training pipeline
├── requirements.txt     # Python dependencies
├── README.md
├── data/
│   ├── telco_churn.csv  # Generated dataset (7,043 records)
│   └── model_results.csv
├── models/
│   └── churn_model.pkl  # Trained model + scaler + encoders
├── templates/
│   ├── index.html       # Prediction UI
│   └── analytics.html   # Analytics dashboard
└── static/
    ├── css/style.css    # Premium dark stylesheet
    ├── js/main.js       # Prediction interactivity
    └── img/             # Generated charts (7 PNG files)
```

## 🤖 Models Trained

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~79% | ~72% |
| Decision Tree | ~79% | ~71% |
| Random Forest | ~79% | ~68% |
| SVM (RBF) | ~80% | ~65% |
| Gradient Boosting | ~79% | ~72% |

## 🌐 Web Application Pages

- **`/`** — Customer profile form + live churn probability gauge
- **`/analytics`** — 7 charts + model comparison table + pipeline methodology
- **`/api/results`** — JSON model metrics
- **`/api/feature_importance`** — JSON feature importance scores

## 🛠 Tech Stack

Python · Flask · Scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · Bootstrap 5 · Git

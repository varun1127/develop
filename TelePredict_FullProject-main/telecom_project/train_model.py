"""
Telecom Churn Prediction – Training Pipeline
Run this ONCE before starting the Flask app:  python train_model.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

os.makedirs("data",       exist_ok=True)
os.makedirs("models",     exist_ok=True)
os.makedirs("static/img", exist_ok=True)

# ── Colour palette for charts ────────────────────────────────────────────────
BG      = "#0A0E1A"
SURFACE = "#111827"
ACCENT  = "#6366F1"
GREEN   = "#10B981"
RED     = "#EF4444"
AMBER   = "#F59E0B"
PURPLE  = "#8B5CF6"
CYAN    = "#06B6D4"
TEXT    = "#E2E8F0"
MUTED   = "#64748B"

COLORS5 = [ACCENT, GREEN, AMBER, RED, PURPLE]

def style_ax(ax, title=""):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values(): sp.set_edgecolor("#1E293B")
    if title: ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

# ── 1. Generate Dataset ──────────────────────────────────────────────────────
print("=" * 60)
print("  TELECOM CHURN PREDICTION — TRAINING PIPELINE")
print("=" * 60)
print("\n[1/8] Generating synthetic Telco dataset…")
np.random.seed(42)
N = 7043

df = pd.DataFrame()
df["customerID"]       = [f"CUST-{i:05d}" for i in range(1, N+1)]
df["gender"]           = np.random.choice(["Male","Female"], N)
df["SeniorCitizen"]    = np.random.choice([0,1], N, p=[0.84,0.16])
df["Partner"]          = np.random.choice(["Yes","No"], N)
df["Dependents"]       = np.random.choice(["Yes","No"], N, p=[0.30,0.70])
df["tenure"]           = np.random.randint(0, 72, N)
df["PhoneService"]     = np.random.choice(["Yes","No"], N, p=[0.90,0.10])
df["MultipleLines"]    = np.where(df["PhoneService"]=="No","No phone service",
                            np.random.choice(["Yes","No"], N))
df["InternetService"]  = np.random.choice(["DSL","Fiber optic","No"], N, p=[0.34,0.44,0.22])
for col in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    df[col] = np.where(df["InternetService"]=="No","No internet service",
                       np.random.choice(["Yes","No"], N))
df["Contract"]         = np.random.choice(["Month-to-month","One year","Two year"], N, p=[0.55,0.21,0.24])
df["PaperlessBilling"] = np.random.choice(["Yes","No"], N, p=[0.59,0.41])
df["PaymentMethod"]    = np.random.choice(["Electronic check","Mailed check",
                            "Bank transfer (automatic)","Credit card (automatic)"], N)
df["MonthlyCharges"]   = np.round(np.random.uniform(18, 118, N), 2)
df["TotalCharges"]     = np.round(df["tenure"] * df["MonthlyCharges"]
                            * np.random.uniform(0.9, 1.1, N), 2)

churn_p = (
    0.05
    + 0.28 * (df["Contract"]=="Month-to-month").astype(float)
    + 0.12 * (df["InternetService"]=="Fiber optic").astype(float)
    + 0.10 * (df["tenure"] < 12).astype(float)
    + 0.06 * (df["SeniorCitizen"]==1).astype(float)
    - 0.12 * (df["tenure"] > 36).astype(float)
    - 0.10 * (df["Contract"]=="Two year").astype(float)
    + np.random.normal(0, 0.05, N)
)
df["Churn"] = np.where(np.random.random(N) < np.clip(churn_p, 0.02, 0.95), "Yes","No")
df.to_csv("data/telco_churn.csv", index=False)
print(f"  ✓ {df.shape[0]:,} rows | {df.shape[1]} columns | "
      f"Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")

# ── 2. Clean ─────────────────────────────────────────────────────────────────
print("\n[2/8] Cleaning & encoding…")
df2 = df.copy()
df2["TotalCharges"] = pd.to_numeric(df2["TotalCharges"], errors="coerce")
df2["TotalCharges"].fillna(df2["TotalCharges"].median(), inplace=True)
df2.drop("customerID", axis=1, inplace=True)
df2["Churn"] = (df2["Churn"]=="Yes").astype(int)

label_encoders = {}
for col in df2.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col])
    label_encoders[col] = le
print(f"  ✓ Encoded {len(label_encoders)} categorical columns")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("\n[3/8] Feature engineering…")
df2["ChargesPerMonth"] = df2["TotalCharges"] / (df2["tenure"] + 1)
df2["TenureGroup"]     = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                labels=[0,1,2,3]).astype(float).fillna(0).astype(int)
df2["NumServices"]     = (
    (df["PhoneService"]=="Yes").astype(int) +
    (df["InternetService"]!="No").astype(int) +
    (df["OnlineSecurity"]=="Yes").astype(int) +
    (df["OnlineBackup"]=="Yes").astype(int) +
    (df["DeviceProtection"]=="Yes").astype(int) +
    (df["TechSupport"]=="Yes").astype(int) +
    (df["StreamingTV"]=="Yes").astype(int) +
    (df["StreamingMovies"]=="Yes").astype(int)
)
print("  ✓ ChargesPerMonth | TenureGroup | NumServices")

# ── 4. Split ──────────────────────────────────────────────────────────────────
print("\n[4/8] Train/test split (80/20)…")
X = df2.drop("Churn", axis=1)
y = df2["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                           random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)
print(f"  ✓ Train: {len(X_tr):,} | Test: {len(X_te):,}")

# ── 5. Train ──────────────────────────────────────────────────────────────────
print("\n[5/8] Training 5 models…")
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}
SCALED = {"Logistic Regression", "SVM"}

results, trained = {}, {}
for name, model in MODELS.items():
    Xtr = X_tr_sc if name in SCALED else X_tr
    Xte = X_te_sc  if name in SCALED else X_te
    model.fit(Xtr, y_tr)
    yp   = model.predict(Xte)
    ypr  = model.predict_proba(Xte)[:, 1]
    results[name] = {
        "Accuracy":  round(accuracy_score(y_te, yp)*100, 2),
        "Precision": round(precision_score(y_te, yp)*100, 2),
        "Recall":    round(recall_score(y_te, yp)*100, 2),
        "F1 Score":  round(f1_score(y_te, yp)*100, 2),
        "ROC-AUC":   round(roc_auc_score(y_te, ypr)*100, 2),
    }
    trained[name] = model
    print(f"  ✓ {name:<25} Acc {results[name]['Accuracy']}%  "
          f"AUC {results[name]['ROC-AUC']}%")

# ── 6. Visualisations ─────────────────────────────────────────────────────────
print("\n[6/8] Generating charts…")
res_df = pd.DataFrame(results).T

# — Model comparison —
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); style_ax(ax, "Model Performance Comparison")
x = np.arange(len(res_df))
w = 0.14
metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
for i, (m, c) in enumerate(zip(metrics, COLORS5)):
    ax.bar(x + i*w, res_df[m], w, label=m, color=c, alpha=0.87, edgecolor="none")
ax.set_xticks(x + w*2); ax.set_xticklabels(res_df.index, rotation=12, ha="right", color=TEXT)
ax.legend(fontsize=9, facecolor=SURFACE, labelcolor=TEXT, edgecolor="#1E293B")
ax.set_ylim(0, 108); ax.set_ylabel("Score (%)", color=MUTED); ax.grid(axis="y", alpha=0.15, color=MUTED)
plt.tight_layout(); plt.savefig("static/img/model_comparison.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — ROC curves —
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG); style_ax(ax, "ROC Curves — All Models")
for (name, model), color in zip(trained.items(), COLORS5):
    Xte = X_te_sc if name in SCALED else X_te
    fpr, tpr, _ = roc_curve(y_te, model.predict_proba(Xte)[:,1])
    auc = results[name]["ROC-AUC"]
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc}%)")
ax.plot([0,1],[0,1],"--", color=MUTED, lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9, facecolor=SURFACE, labelcolor=TEXT, edgecolor="#1E293B")
ax.grid(alpha=0.12, color=MUTED)
plt.tight_layout(); plt.savefig("static/img/roc_curves.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — Confusion matrix (best AUC model) —
best_name = res_df["ROC-AUC"].idxmax()
best_m    = trained[best_name]
Xte_b = X_te_sc if best_name in SCALED else X_te
cm = confusion_matrix(y_te, best_m.predict(Xte_b))
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Not Churn","Churn"], yticklabels=["Not Churn","Churn"],
            annot_kws={"size":14,"color":"white","weight":"bold"}, linewidths=0.5, linecolor=BG)
ax.set_title(f"Confusion Matrix — {best_name}", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax.tick_params(colors=TEXT); ax.set_xlabel("Predicted", color=MUTED); ax.set_ylabel("Actual", color=MUTED)
plt.tight_layout(); plt.savefig("static/img/confusion_matrix.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — Feature importance —
rf   = trained["Random Forest"]
fi   = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG); style_ax(ax, "Top 15 Feature Importances (Random Forest)")
bar_colors = [RED if v>0.10 else AMBER if v>0.05 else ACCENT for v in fi.values]
fi.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="none")
ax.set_xticklabels(fi.index, rotation=40, ha="right", color=TEXT, fontsize=9)
ax.set_ylabel("Importance"); ax.grid(axis="y", alpha=0.15, color=MUTED)
plt.tight_layout(); plt.savefig("static/img/feature_importance.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — Churn analysis (distribution + tenure) —
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
cc = (df["Churn"]=="Yes").sum(); cr = (df["Churn"]=="No").sum()
axes[0].pie([cr, cc], labels=["Retained","Churned"], colors=[GREEN, RED],
            autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":BG,"linewidth":2},
            textprops={"color":TEXT})
axes[0].set_title("Churn Distribution", color=TEXT, fontsize=12, fontweight="bold")
axes[0].set_facecolor(BG)
tenure_cr = df.groupby("tenure")["Churn"].apply(lambda x:(x=="Yes").mean()*100)
axes[1].plot(tenure_cr.index, tenure_cr.values, color=RED, lw=2)
axes[1].fill_between(tenure_cr.index, tenure_cr.values, alpha=0.2, color=RED)
style_ax(axes[1], "Churn Rate by Tenure (months)")
axes[1].set_xlabel("Tenure (months)"); axes[1].set_ylabel("Churn Rate (%)")
axes[1].grid(alpha=0.12, color=MUTED)
plt.tight_layout(); plt.savefig("static/img/churn_analysis.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — Heatmap: Contract × Internet —
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(SURFACE)
pivot = df.groupby(["Contract","InternetService"])["Churn"].apply(
    lambda x:(x=="Yes").mean()*100).unstack()
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax,
            linewidths=0.5, linecolor=BG, annot_kws={"size":12,"weight":"bold"})
ax.set_title("Churn Rate: Contract × Internet Service (%)", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax.tick_params(colors=TEXT)
plt.tight_layout(); plt.savefig("static/img/churn_heatmap.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()

# — Monthly charges boxplot —
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
ret = df[df["Churn"]=="No"]["MonthlyCharges"]
chu = df[df["Churn"]=="Yes"]["MonthlyCharges"]
bp = axes[0].boxplot([ret, chu], patch_artist=True,
    tick_labels=["Retained","Churned"], widths=0.5,
    medianprops=dict(color="white", lw=2),
    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
    flierprops=dict(marker="o", color=MUTED, alpha=0.4, markersize=3))
bp["boxes"][0].set_facecolor(GREEN+"55"); bp["boxes"][0].set_edgecolor(GREEN)
bp["boxes"][1].set_facecolor(RED+"55");   bp["boxes"][1].set_edgecolor(RED)
style_ax(axes[0], "Monthly Charges by Churn Status")
axes[0].set_ylabel("Monthly Charges ($)"); axes[0].grid(axis="y", alpha=0.15, color=MUTED)
parts = axes[1].violinplot([ret, chu], positions=[1,2], showmeans=True, showmedians=True)
for pc, c in zip(parts["bodies"], [GREEN, RED]): pc.set_facecolor(c); pc.set_alpha(0.55)
parts["cmeans"].set_color("white"); parts["cmedians"].set_color(AMBER)
for k in ["cbars","cmaxes","cmins"]: parts[k].set_color(MUTED)
axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["Retained","Churned"], color=TEXT)
style_ax(axes[1], "Violin Plot — Monthly Charges")
axes[1].set_ylabel("Monthly Charges ($)"); axes[1].grid(axis="y", alpha=0.15, color=MUTED)
plt.tight_layout(); plt.savefig("static/img/charges_dist.png", dpi=130, bbox_inches="tight", facecolor=BG); plt.close()
print("  ✓ 7 charts saved to static/img/")

# ── 7. Save Artefacts ─────────────────────────────────────────────────────────
print("\n[7/8] Saving model artefacts…")
pd.DataFrame(results).T.to_csv("data/model_results.csv")
with open("models/churn_model.pkl","wb") as f:
    pickle.dump({
        "model":          trained["Random Forest"],
        "all_models":     trained,
        "scaler":         scaler,
        "label_encoders": label_encoders,
        "feature_names":  list(X.columns),
        "results":        results,
        "best_name":      best_name,
        "churn_rate":     float((df["Churn"]=="Yes").mean()),
        "n_samples":      N,
    }, f)
print(f"  ✓ Best model by AUC: {best_name}")
print("  ✓ Saved: models/churn_model.pkl")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n[8/8] Results summary")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-" * 70)
for n, m in results.items():
    print(f"{n:<25} {m['Accuracy']:>8}% {m['Precision']:>9}% "
          f"{m['Recall']:>7}% {m['F1 Score']:>7}% {m['ROC-AUC']:>7}%")
print("=" * 70)
print(f"\n✅  Training complete!  Best AUC → {best_name}")
print("    Run `python app.py` to launch the web application.\n")

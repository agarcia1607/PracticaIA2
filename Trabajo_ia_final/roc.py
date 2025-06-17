import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# ------------------- Ajustes de rutas -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "df_v8.csv")
OUT_DIR = os.path.join(BASE_DIR, "figuras_roc_pr")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- Carga y reconstrucción -------------------
df = pd.read_csv(CSV_PATH)
dummy_cols = [c for c in df.columns if c.startswith("alcohol_level_")]
df["alcohol_level"] = "bajo"
for col in dummy_cols:
    nivel = col.replace("alcohol_level_", "")
    df.loc[df[col] == 1, "alcohol_level"] = nivel

X = df.drop(columns=dummy_cols + ["alcohol_level"])
y = df["alcohol_level"]

# --------------- Escalado y split ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ------------ Binarización multiclass -----------------
lb = LabelBinarizer()
Y_train = lb.fit_transform(y_train)
Y_test  = lb.transform(y_test)

# ------------------ Modelos ---------------------------
models = {
    "Árbol": DecisionTreeClassifier(criterion="gini", random_state=42),
    "KNN":    KNeighborsClassifier(n_neighbors=5),
    "SVM":    SVC(kernel="rbf", probability=True, random_state=42),
    "RedNN":  MLPClassifier(hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-3,
                            max_iter=200,
                            random_state=42)
}

# ----------- Entrenamiento y probabilidades -----------
y_score = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_score[name] = model.predict_proba(X_test)

# --------------- Curva ROC (micro-average) -------------
plt.figure(figsize=(8,6))
for name, proba in y_score.items():
    fpr, tpr, _ = roc_curve(Y_test.ravel(), proba.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], linestyle="--", color="gray", label="Aleatorio")
plt.title("Curvas ROC (micro-average) - Caso 8")
plt.xlabel("Tasa Falsos Positivos")
plt.ylabel("Tasa Verdaderos Positivos")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_micro_caso8.png"))
plt.close()

# -------- Curva Precision–Recall (micro-average) -------
plt.figure(figsize=(8,6))
for name, proba in y_score.items():
    precision, recall, _ = precision_recall_curve(Y_test.ravel(), proba.ravel())
    ap = average_precision_score(Y_test, proba, average="micro")
    plt.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.2f})")

plt.title("Curvas Precision–Recall (micro-average) - Caso 8")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_micro_caso8.png"))
plt.close()

print(f"✅ Gráficas guardadas en {OUT_DIR}")



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ──────────────────────────────────────────────────────────────
BASE_DIR   = os.getcwd()                            # de donde ejecutes
OUTPUT_DIR = os.path.join(BASE_DIR, "figuras_modelos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

datasets = [f"df_v{i}.csv" for i in range(1, 9)]
# ────────────────────────────────────────────────────────────────────────

for fname in datasets:
    path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️ {fname} no encontrado, se salta.")
        continue

    df = pd.read_csv(path)

    # ── Reconstruir etiqueta ─────────────────────────────────────────────
    df['alcohol_level'] = 'bajo'
    if 'alcohol_level_medio'    in df: df.loc[df['alcohol_level_medio']==1,    'alcohol_level'] = 'medio'
    if 'alcohol_level_alto'      in df: df.loc[df['alcohol_level_alto']==1,      'alcohol_level'] = 'alto'
    if 'alcohol_level_muy_alto'  in df: df.loc[df['alcohol_level_muy_alto']==1,  'alcohol_level'] = 'muy_alto'

    # ── Features y target ─────────────────────────────────────────────────
    dummy_cols = [c for c in df.columns if c.startswith("alcohol_level_")]
    X = df.drop(columns=dummy_cols + ['alcohol_level'])
    y = df['alcohol_level']

    # ── Encode target para gráficas ───────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train/Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    # ── 1) Árbol de Decisión ─────────────────────────────────────────────
    tree = DecisionTreeClassifier(criterion='gini', random_state=42)
    tree.fit(X_train, y_train)

    # Diagrama del árbol
    plt.figure(figsize=(14,10))
    plot_tree(tree,
              feature_names=X.columns,
              class_names=le.classes_,
              filled=True,
              fontsize=6)
    plt.title(f"Árbol de Decisión — {fname}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{fname[:-4]}_arbol.png"))
    plt.close()

    # Importancias
    imp = pd.Series(tree.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    sns.barplot(x=imp.values, y=imp.index)
    plt.title(f"Importancia de Features — {fname}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{fname[:-4]}_importancias.png"))
    plt.close()

    # ── 2) KNN + PCA Scatter ─────────────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(7,6))
    plt.scatter(X_test_pca[:,0], X_test_pca[:,1],
                c=knn.predict(X_test),
                cmap='Set1', alpha=0.7, edgecolor='k')
    plt.title(f"KNN (k=5) — {fname}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{fname[:-4]}_knn.png"))
    plt.close()

    # ── 3) SVM — Regiones de Decisión sobre PCA ──────────────────────────
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)

    # Crear malla sobre el espacio PCA
    x_min, x_max = X_test_pca[:,0].min()-1, X_test_pca[:,0].max()+1
    y_min, y_max = X_test_pca[:,1].min()-1, X_test_pca[:,1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Invertir PCA para volver al espacio original
    grid = np.c_[xx.ravel(), yy.ravel()]
    inv = pca.inverse_transform(grid)
    Z = svm.predict(inv).reshape(xx.shape)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    plt.scatter(X_test_pca[:,0], X_test_pca[:,1],
                c=y_test, cmap='Set1', edgecolor='k')
    plt.title(f"SVM — {fname}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{fname[:-4]}_svm.png"))
    plt.close()

    print(f"✅ Figuras para {fname} generadas.")

print("🎉 Todas las figuras están en:", OUTPUT_DIR)

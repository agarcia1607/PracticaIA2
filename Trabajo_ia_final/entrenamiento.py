import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn

# 📁 Carpeta donde están tus df_v*.csv
BASE_DIR = os.getcwd()

# 🧠 Arquitectura de la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.modelo(x)

# 📊 Para almacenar resultados
resultados = []

def entrenar_y_evaluar(df, nombre):
    # 1) Reconstruir la columna 'alcohol_level' a partir de dummies
    #    Asumimos dummies: alcohol_level_medio, alcohol_level_alto, alcohol_level_muy_alto
    df = df.copy()
    # Inicializamos todos como 'bajo'
    df['alcohol_level'] = 'bajo'
    if 'alcohol_level_medio' in df.columns:
        df.loc[df['alcohol_level_medio'] == 1, 'alcohol_level'] = 'medio'
    if 'alcohol_level_alto' in df.columns:
        df.loc[df['alcohol_level_alto'] == 1, 'alcohol_level'] = 'alto'
    if 'alcohol_level_muy_alto' in df.columns:
        df.loc[df['alcohol_level_muy_alto'] == 1, 'alcohol_level'] = 'muy_alto'

    # 2) Features X y target y
    dummy_cols = [c for c in df.columns if c.startswith('alcohol_level_')]
    X = df.drop(columns=dummy_cols + ['alcohol_level'])
    y = df['alcohol_level']

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4) Modelos clásicos
    modelos = {
        'Arbol': DecisionTreeClassifier(),
        'KNN':   KNeighborsClassifier(),
        'SVM':   SVC()
    }
    for nombre_modelo, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        rep = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        resultados.append({
            'dataset': nombre,
            'modelo': nombre_modelo,
            'accuracy': rep['accuracy'],
            'precision': rep['weighted avg']['precision'],
            'recall':    rep['weighted avg']['recall'],
            'f1-score':  rep['weighted avg']['f1-score']
        })

    # 5) Red Neuronal
    # Convertir etiquetas a códigos
    y_train_codes = y_train.astype('category').cat.codes
    y_test_codes  = y_test.astype('category').cat.codes
    input_dim  = X_train.shape[1]
    output_dim = y_train_codes.nunique()

    # Tensores
    Xtr = torch.tensor(X_train.values, dtype=torch.float32)
    Ytr = torch.tensor(y_train_codes.values, dtype=torch.long)
    Xte = torch.tensor(X_test.values,  dtype=torch.float32)
    Yte = torch.tensor(y_test_codes.values,  dtype=torch.long)

    net = RedNeuronal(input_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(net.parameters(), lr=0.01)

    # Entrenamos 100 epochs
    for epoch in range(100):
        net.train()
        opt.zero_grad()
        logits = net(Xtr)
        loss   = loss_fn(logits, Ytr)
        loss.backward()
        opt.step()

    # Evaluación
    net.eval()
    with torch.no_grad():
        preds = net(Xte).argmax(dim=1)
        rep_nn = classification_report(
            Yte, preds, output_dict=True, zero_division=0
        )
        resultados.append({
            'dataset': nombre,
            'modelo': 'Red Neuronal',
            'accuracy': rep_nn['accuracy'],
            'precision': rep_nn['weighted avg']['precision'],
            'recall':    rep_nn['weighted avg']['recall'],
            'f1-score':  rep_nn['weighted avg']['f1-score']
        })

# 🚀 Ejecutar para df_v1 … df_v8
for i in range(1, 9):
    nombre = f"df_v{i}"
    ruta = os.path.join(BASE_DIR, f"{nombre}.csv")
    if os.path.exists(ruta):
        print(f"Procesando {nombre} …")
        df_ = pd.read_csv(ruta)
        entrenar_y_evaluar(df_, nombre)
    else:
        print(f"⚠️ {nombre}.csv no se encontró; se omite.")

# 💾 Guardar resultados
pd.DataFrame(resultados).to_csv(
    os.path.join(BASE_DIR, "resultados_modelos_resumen.csv"),
    index=False
)
print("✅ resultados_modelos_resumen.csv generado.")

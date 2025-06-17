import pandas as pd
import matplotlib.pyplot as plt
import os

# Ajusta esta ruta si hace falta
CSV_PATH = "resultados_modelos_resumen.csv"
OUTPUT_DIR = "graficas_f1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargamos los resultados
df = pd.read_csv(CSV_PATH)

# Extraemos la variable de preprocesamiento a partir del nombre del dataset
df["normalizado"] = df["dataset"].isin(["df_v5", "df_v6", "df_v7", "df_v8"])
df["outliers"]     = df["dataset"].isin(["df_v3", "df_v4", "df_v7", "df_v8"])
df["balanceado"]   = df["dataset"].isin(["df_v2", "df_v4", "df_v6", "df_v8"])

# 1) F1-score máximo por algoritmo (general)
max_f1 = df.groupby("modelo")["f1-score"].max()
plt.figure()
max_f1.plot(kind="bar")
plt.title("F1-Score Máximo por Algoritmo")
plt.ylabel("F1-Score Máximo")
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_maximo_por_algoritmo.png"))
plt.close()

# 2) F1-score medio por normalización y algoritmo
mean_norm = df.groupby(["normalizado","modelo"])["f1-score"].mean().unstack()
plt.figure()
mean_norm.plot(kind="bar")
plt.title("F1-Score Medio por Normalización y Algoritmo")
plt.ylabel("F1-Score Medio")
plt.ylim(0,1.05)
plt.xlabel("¿Dataset Normalizado?")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_medio_por_normalizacion.png"))
plt.close()

# 3) F1-score medio por manejo de outliers y algoritmo
mean_out = df.groupby(["outliers","modelo"])["f1-score"].mean().unstack()
plt.figure()
mean_out.plot(kind="bar")
plt.title("F1-Score Medio por Eliminación de Outliers y Algoritmo")
plt.ylabel("F1-Score Medio")
plt.ylim(0,1.05)
plt.xlabel("¿Se Eliminan Outliers?")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_medio_por_outliers.png"))
plt.close()

# 4) F1-score medio por balanceo y algoritmo
mean_bal = df.groupby(["balanceado","modelo"])["f1-score"].mean().unstack()
plt.figure()
mean_bal.plot(kind="bar")
plt.title("F1-Score Medio por Balanceo y Algoritmo")
plt.ylabel("F1-Score Medio")
plt.ylim(0,1.05)
plt.xlabel("¿Dataset Balanceado?")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_medio_por_balanceo.png"))
plt.close()

print(f"Gráficas guardadas en carpeta «{OUTPUT_DIR}».")

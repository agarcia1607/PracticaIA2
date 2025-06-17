import pandas as pd
import os

# Paso 1: Descargar el dataset
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(URL, sep=';')
print("✅ Dataset descargado con éxito. Forma:", df.shape)

# Paso 2: Crear la variable categórica basada en el alcohol
df["alcohol_level"] = pd.cut(
    df["alcohol"],
    bins=[0, 9, 11, 13, df["alcohol"].max() + 0.1],
    labels=["bajo", "medio", "alto", "muy_alto"],
    include_lowest=True
)

# Paso 3: Eliminar la columna original de alcohol para evitar fuga de información
df_final = df.drop(columns=["alcohol"])

# Paso 4: Guardar dataset procesado
output_path = os.path.join(os.getcwd(), "wine_alcohol_levels.csv")
df_final.to_csv(output_path, index=False)
print(f"✅ Dataset procesado y guardado en: {output_path}")

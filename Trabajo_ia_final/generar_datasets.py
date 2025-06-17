import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy.stats import zscore

# Usar el directorio actual del script para guardar los CSV
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cargar_datos():
    df = pd.read_csv(os.path.join(BASE_DIR, "wine_alcohol_levels.csv"), sep=",")
    df.columns = df.columns.str.strip()
    return df


def codificar_categoricas(df):
    if "alcohol_level" in df.columns:
        df = pd.get_dummies(df, columns=["alcohol_level"], drop_first=True)
    return df


def escalar_datos(df):
    df = df.copy()
    columnas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                'density', 'pH', 'sulphates']
    scaler = MinMaxScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df


def mantener_outliers(df, columnas, porcentaje_outliers=0.05):
    df = df.copy()
    df_muy_alto = df[df['alcohol_level'] == 'muy_alto']
    df_restantes = df[df['alcohol_level'] != 'muy_alto']

    z_scores = np.abs(zscore(df_restantes[columnas]))
    outlier_mask = (z_scores > 2.5).any(axis=1)
    num_outliers = int(len(df) * porcentaje_outliers)
    outliers = df_restantes[outlier_mask].sample(n=min(num_outliers, outlier_mask.sum()), random_state=42)
    no_outliers = df_restantes[~outlier_mask]

    df_out = pd.concat([no_outliers, outliers, df_muy_alto]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_out


def balancear_clases(df):
    clases = df["alcohol_level"].dropna().unique()
    grupos = [df[df["alcohol_level"] == c] for c in clases]
    if not grupos:
        raise ValueError("No hay grupos para balancear.")
    max_len = max(len(g) for g in grupos if len(g) > 0)
    balanceado = [resample(g, replace=True, n_samples=max_len, random_state=42) for g in grupos if len(g) > 0]
    return pd.concat(balanceado).sample(frac=1, random_state=42).reset_index(drop=True)


def generar_y_guardar_version(nombre, base_df, escalar=False, outliers=False, balancear=False):
    ruta_salida = os.path.join(BASE_DIR, f"{nombre}.csv")
    if os.path.exists(ruta_salida):
        print(f"{nombre}.csv ya existe. Omitiendo...")
        return

    df = base_df.copy()

    if outliers:
        columnas_numericas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                              'density', 'pH', 'sulphates']
        df = mantener_outliers(df, columnas_numericas, porcentaje_outliers=0.05)

    if balancear:
        df = balancear_clases(df)

    df = codificar_categoricas(df)

    if escalar:
        df = escalar_datos(df)

    df.to_csv(ruta_salida, index=False)
    print(f"{nombre}.csv generado y guardado.")


def main():
    base_df = cargar_datos()

    configuraciones = {
        "df_v1": {"escalar": False, "outliers": False, "balancear": False},
        "df_v2": {"escalar": False, "outliers": False, "balancear": True},
        "df_v3": {"escalar": False, "outliers": True,  "balancear": False},
        "df_v4": {"escalar": False, "outliers": True,  "balancear": True},
        "df_v5": {"escalar": True,  "outliers": False, "balancear": False},
        "df_v6": {"escalar": True,  "outliers": False, "balancear": True},
        "df_v7": {"escalar": True,  "outliers": True,  "balancear": False},
        "df_v8": {"escalar": True,  "outliers": True,  "balancear": True},
    }

    for nombre, params in configuraciones.items():
        generar_y_guardar_version(nombre, base_df, **params)


if __name__ == "__main__":
    main()

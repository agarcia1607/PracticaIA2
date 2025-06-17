
import pandas as pd

# Cargar dataset local completo
archivo_local = "wine.csv"  # Asegúrate que el archivo esté en la misma carpeta
df = pd.read_csv(archivo_local, sep=';', quotechar='"')

# Asegurarse que los nombres de columnas estén limpios
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Mostrar info básica para verificar
print("Primeras filas del dataset:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)



# %% [markdown]
# # 3. Aprendizaje No Supervisado
# ## 3.1 Dataset y preprocesado

# %%

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import zscore


# Eliminar variable objetivo
X = df.drop('alcohol', axis=1)

print("Valores nulos por columna antes de limpiar:")
print(df.isnull().sum())

# Verificar
print("Forma del dataset sin variable objetivo:", X.shape)
print("Columnas:", X.columns.tolist())

# %% [markdown]
# ### Definimos funciones de Preprocesamiento

# %%
def normalizar_datos(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def reducir_dimensiones(X, n_componentes=2):
    pca = PCA(n_components=n_componentes)
    return pd.DataFrame(pca.fit_transform(X), columns=[f'PC{i+1}' for i in range(n_componentes)])

def eliminar_outliers(X):
    z_scores = np.abs(zscore(X))
    umbral = 2.0
    mascara = (z_scores < umbral).all(axis=1)
    return X[mascara].reset_index(drop=True)

def balancear_datos(X):
    kmeans = KMeans(n_clusters=7, random_state=42)  
    etiquetas = kmeans.fit_predict(X)
    conteo = np.bincount(etiquetas)
    tamano_min = conteo[conteo > 0].min()  
    X_balanceado = pd.DataFrame()
    for etiqueta in np.unique(etiquetas):
        idx = np.where(etiquetas == etiqueta)[0]
        if len(idx) >= tamano_min:
            indices_muestreados = np.random.choice(idx, size=tamano_min, replace=False)
            X_balanceado = pd.concat([X_balanceado, X.iloc[indices_muestreados]], axis=0)
    return X_balanceado.reset_index(drop=True)

# %% [markdown]
# ### Generamos las Ocho Versiones del Dataset 
# Creamos las ocho configuraciones según la Figura 2, aplicando las combinaciones de preprocesamiento.

# %%
datasets = {}
nombres_config = [
    "CC(SI)_ED(NO)_Outliers(NO)_Balanceo(NO)",
    "CC(SI)_ED(NO)_Outliers(NO)_Balanceo(SI)",
    "CC(SI)_ED(NO)_Outliers(SI)_Balanceo(NO)",
    "CC(SI)_ED(NO)_Outliers(SI)_Balanceo(SI)",
    "CC(SI)_ED(SI)_Outliers(NO)_Balanceo(NO)",
    "CC(SI)_ED(SI)_Outliers(NO)_Balanceo(SI)",
    "CC(SI)_ED(SI)_Outliers(SI)_Balanceo(NO)",
    "CC(SI)_ED(SI)_Outliers(SI)_Balanceo(SI)"
]

for config in nombres_config:
    X_temp = X.copy()
    
    # Normalización (siempre SÍ)
    X_temp = normalizar_datos(X_temp)
    
    # Reducción de dimensionalidad
    if "ED(SI)" in config:
        X_temp = reducir_dimensiones(X_temp)
    
    # Eliminación de outliers
    if "Outliers(SI)" in config:
        X_temp = eliminar_outliers(X_temp)
    
    # Balanceo
    if "Balanceo(SI)" in config:
        X_temp = balancear_datos(X_temp)
    
    datasets[config] = X_temp
    print(f"Forma del dataset {config}: {X_temp.shape}")

# %% [markdown]
# ## 3.2 Entrenamiento con K-Means y DBSCAN
# 
# Entrenamos K-Means para cada dataset, determinando el número óptimo de clústeres con el método del codo y la puntuación de silhouette.

# %%
def evaluar_kmeans(X, max_k=10):
    inercias = []
    silhouettes = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        if len(np.unique(kmeans.labels_)) > 1:
            silhouettes.append(silhouette_score(X, kmeans.labels_))
        else:
            silhouettes.append(-1)
    # Seleccionar k óptimo
    k_optimo = K[np.argmax(silhouettes)]
    return k_optimo

# Entrenar K-Means
resultados_kmeans = {}
for config, X_temp in datasets.items():
    print(f"\nEntrenando K-Means para {config}")
    k_optimo = evaluar_kmeans(X_temp)
    kmeans = KMeans(n_clusters=k_optimo, random_state=42)
    kmeans.fit(X_temp)
    etiquetas = kmeans.labels_
    inercia = kmeans.inertia_
    silhouette = silhouette_score(X_temp, etiquetas) if len(np.unique(etiquetas)) > 1 else -1
    resultados_kmeans[config] = {
        'modelo': kmeans,
        'k_optimo': k_optimo,
        'inercia': inercia,
        'silhouette': silhouette,
        'etiquetas': etiquetas
    }
    print(f"K óptimo: {k_optimo}, Inercia: {inercia:.2f}, Silhouette: {silhouette:.2f}")

# %% [markdown]
# ## DBSCAN: Ajuste de MinPts y ε
# 
# Ajustamos DBSCAN usando un gráfico de k-distancias para determinar eps y min_samples.

# %%
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def ajustar_dbscan(X):
    min_pts = 2 * X.shape[1]  
    vecinos = NearestNeighbors(n_neighbors=min_pts)
    vecinos_fit = vecinos.fit(X)
    distancias, _ = vecinos_fit.kneighbors(X)
    distancias = np.sort(distancias[:, min_pts-1], axis=0)
    
    # Probar valores de eps
    eps_valores = np.arange(0.5, 5.0, 0.5)  
    mejor_silhouette = -1
    mejores_params = None
    mejores_etiquetas = None
    
    for eps in eps_valores:
        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        etiquetas = dbscan.fit_predict(X)
        if len(np.unique(etiquetas)) > 1 and -1 in etiquetas:
            try:
                silhouette = silhouette_score(X[etiquetas != -1], etiquetas[etiquetas != -1])
                if silhouette > mejor_silhouette:
                    mejor_silhouette = silhouette
                    mejores_params = (eps, min_pts)
                    mejores_etiquetas = etiquetas
            except:
                continue
    
    if mejores_params is None:
        print("No se encontraron clústeres válidos para DBSCAN.")
        return (0.5, min_pts), np.zeros(X.shape[0]), -1
    
    return mejores_params, mejores_etiquetas, mejor_silhouette

# Entrenar DBSCAN
resultados_dbscan = {}
for config, X_temp in datasets.items():
    print(f"\nEntrenando DBSCAN para {config}")
    (eps, min_pts), etiquetas, silhouette = ajustar_dbscan(X_temp)
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    dbscan.fit(X_temp)
    resultados_dbscan[config] = {
        'modelo': dbscan,
        'eps': eps,
        'min_pts': min_pts,
        'silhouette': silhouette,
        'etiquetas': etiquetas
    }
    print(f"Eps óptimo: {eps:.2f}, MinPts: {min_pts}, Silhouette: {silhouette:.2f}")

# %% [markdown]
# ### Generar Tres Casos de Prueba por Técnica
# 
# Generamos tres casos de prueba para cada dataset y algoritmo, prediciendo clústeres para puntos muestreados

# %%
def generar_casos_prueba(X, modelo, n_muestras=3):
    np.random.seed(42)
    indices = np.random.choice(X.index, size=n_muestras, replace=False)
    muestras_prueba = X.iloc[indices]
    if isinstance(modelo, KMeans):
        predicciones = modelo.predict(muestras_prueba)
    else:  # DBSCAN
        predicciones = modelo.fit_predict(muestras_prueba)
    return muestras_prueba, predicciones

# Generar casos de prueba
for config, X_temp in datasets.items():
    print(f"\nCasos de prueba para {config}")
    
    # K-Means
    modelo_kmeans = resultados_kmeans[config]['modelo']
    muestras, predicciones = generar_casos_prueba(X_temp, modelo_kmeans)
    print("Casos de Prueba K-Means:")
    for i, (muestra, pred) in enumerate(zip(muestras.values, predicciones)):
        print(f"Muestra {i+1}: {muestra.round(2)}, Clúster: {pred}")
    
    # DBSCAN
    modelo_dbscan = resultados_dbscan[config]['modelo']
    muestras, predicciones = generar_casos_prueba(X_temp, modelo_dbscan)
    print("Casos de Prueba DBSCAN:")
    for i, (muestra, pred) in enumerate(zip(muestras.values, predicciones)):
        print(f"Muestra {i+1}: {muestra.round(2)}, Clúster: {pred}")

# %% [markdown]
# ## 3.4 Graficas 2D

# %% [markdown]
# ### Grafica para los K-Means

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Crear figura grande con subplots (2 filas x 4 columnas)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()  # Aplanar para iterar fácilmente

# Recorremos los resultados guardados para cada configuración
for idx, (config, resultados) in enumerate(resultados_kmeans.items()):
    kmeans = resultados['modelo']
    X_temp = datasets[config]
    etiquetas = resultados['etiquetas']
    k_optimo = resultados['k_optimo']

    # Aplicar PCA a los datos para reducir a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_temp)

    # Proyectar los centroides al mismo espacio PCA
    centroides_pca = pca.transform(kmeans.cluster_centers_)

    # Graficar en el subplot correspondiente
    ax = axes[idx]
    for cluster_id in range(k_optimo):
        puntos_cluster = X_pca[etiquetas == cluster_id]
        ax.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], label=f'Cluster {cluster_id}', s=30)

    ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
               s=200, c='black', marker='*', label='Centroides')

    ax.set_title(f'Caso {idx + 1}: {config}\nK={k_optimo} ,Silhouette={resultados["silhouette"]:.2f}')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend()
    ax.grid(True)

# Ajustar diseño general
plt.tight_layout()
plt.suptitle('Clusters por configuración (K-Means + PCA)', fontsize=16, y=1.05)
plt.show()


# %% [markdown]
# ### Grafica para DBSCAN

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Crear figura con 8 subgráficos (2 filas, 4 columnas)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Paleta de colores para clusters (excluyendo el ruido)
from matplotlib import cm
cmap = cm.get_cmap('tab10')

for idx, (config, resultados) in enumerate(resultados_dbscan.items()):
    etiquetas = resultados['etiquetas']
    X_temp = datasets[config]

    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_temp)

    # Graficar clusters
    ax = axes[idx]
    etiquetas_unicas = np.unique(etiquetas)

    for etiqueta in etiquetas_unicas:
        puntos = X_pca[etiquetas == etiqueta]
        if etiqueta == -1:
            # Ruido: color gris
            ax.scatter(puntos[:, 0], puntos[:, 1], c='gray', label='Ruido', s=30, marker='x')
        else:
            color = cmap(etiqueta % 10)  # Rotar colores si hay más de 10 clusters
            ax.scatter(puntos[:, 0], puntos[:, 1], c=[color], label=f'Cluster {etiqueta}', s=30)

    ax.set_title(f'Caso {idx + 1}:{config}\nEps={resultados["eps"]:.2f}, Silhouette={resultados["silhouette"]:.2f}')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend()
    ax.grid(True)

# Ajustar espacios entre subgráficos
plt.tight_layout()
plt.suptitle('Visualización DBSCAN + PCA (8 configuraciones)', fontsize=16, y=1.05)
plt.show()




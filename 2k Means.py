import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Carregando o dataset Titanic
titanic = sns.load_dataset('titanic')

# Definindo as variáveis de interesse para o clustering conforme exercício
feature_names = ['pclass', 'age', 'fare']

# Limpar dados não considerados
titanic.dropna(subset=feature_names, inplace=True)

# Aplicar log nas variáveis 'fare' e 'age'
titanic['log_fare'] = np.log(titanic['fare'] + 1)  # log(1 + fare) para evitar problemas de divisão por 0
titanic['log_age'] = np.log(titanic['age'] + 1)  # log(1 + age)

# Atualizando as variáveis de clustering
feature_names = ['pclass', 'log_age', 'log_fare']
X = titanic[feature_names].dropna().to_numpy()

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# redução de dimensionalidade via pca
pca = PCA(n_components=2)  # 2 componentes principais
X_pca = pca.fit_transform(X_scaled)

# Método do cotovelo para escolher o melhor número de clusters (k)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)  # Aplicando K-means nos dados transformados pela PCA
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

# Plotando o método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Método do Cotovelo - Inércia vs Número de Clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.show()

# Plotando o índice de silhueta
plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='green')
plt.title('Índice de Silhueta vs Número de Clusters')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Índice de Silhueta')
plt.grid(True)
plt.show()

# número de clusters (k), baseado nas observações dos gráficos
k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, init='k-means++', n_init=50, max_iter=300, random_state=42)
clusters = kmeans_final.fit_predict(X_pca)  # Aplicando K-means nos dados transformados pela PCA

# Adicionar os clusters ao DataFrame original
titanic['cluster'] = clusters

# 4. Análise dos clusters
print(titanic.groupby('cluster')[feature_names].mean())

# Visualizando a distribuição dos clusters
sns.pairplot(titanic, vars=feature_names, hue='cluster', palette='tab10')
plt.suptitle('Clusters formados com K-means', y=1.02)
plt.show()

# Calcular o índice de silhueta final
sil_score_final = silhouette_score(X_pca, clusters)
print(f'Índice de Silhueta final para k={k_optimal}: {sil_score_final:.4f}')

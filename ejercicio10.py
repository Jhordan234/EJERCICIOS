# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar un conjunto de datos de ejemplo
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Visualizar los datos generados
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Datos generados para K-Means")
plt.show()

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Obtener las etiquetas de los clusters
y_kmeans = kmeans.predict(X)

# Visualizar los clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Mostrar los centros de los clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title("Clusters generados por K-Means")
plt.show()

# Mostrar los centros de los clusters
print("Centros de los clusters: \n", centers)

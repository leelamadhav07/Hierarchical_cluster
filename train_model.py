import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

from sklearn.cluster import AgglomerativeClustering

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/Mall_Customers.csv")

X = df.iloc[:, [3, 4]]

# --------------------------------
# DENDROGRAM
# --------------------------------

plt.figure(figsize=(10, 5))

dendrogram(linkage(X, method="ward"))

plt.title("Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Euclidean Distance")

plt.show()

# --------------------------------
# FINAL MODEL
# --------------------------------

model = AgglomerativeClustering(n_clusters=5)

clusters = model.fit_predict(X)

df["Cluster"] = clusters

print(df.head())

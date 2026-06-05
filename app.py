import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/Mall_Customers.csv")

X = df.iloc[:, [3, 4]]

# --------------------------------
# UI
# --------------------------------

st.title("Hierarchical Clustering")

st.write("Customer Segmentation")

# --------------------------------
# CLUSTER SELECTION
# --------------------------------

n_clusters = st.slider("Select Number of Clusters", 2, 10, 5)

# --------------------------------
# MODEL
# --------------------------------

model = AgglomerativeClustering(n_clusters=n_clusters)

clusters = model.fit_predict(X)

# --------------------------------
# PLOT
# --------------------------------

fig, ax = plt.subplots()

ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

ax.set_xlabel("Annual Income")

ax.set_ylabel("Spending Score")

ax.set_title("Customer Clusters")

st.pyplot(fig)

# --------------------------------
# DATA
# --------------------------------

if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

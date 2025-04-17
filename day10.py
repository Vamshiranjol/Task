##Perform clustering on patient data using K-Means to identify disease patterns.
# Visualize clusters and summarize insights.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
data=pd.read_csv('synthetic_patient_data.csv')

np.random.seed(42)
num_samples=300
data= {
  "Age": np.random.randint(20, 80, num_samples),
    "Fever": np.random.randint(0, 2, num_samples),
    "Cough": np.random.randint(0, 2, num_samples),
    "Fatigue": np.random.randint(0, 2, num_samples),
    "Body Pain": np.random.randint(0, 2, num_samples),
    "Shortness of Breath": np.random.randint(0, 2, num_samples)
}

df=pd.DataFrame(data)
scaler= StandardScaler()
scaled_data= scaler.fit_transform(df)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

##using pca to plot it in 2d 
pca= PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Add PCA results to DataFrame
df["PCA1"] = pca_data[:, 0]
df["PCA2"] = pca_data[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=80)
plt.title("Patient Clusters Identified by K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()






import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Data mahasiswa
students = {
    "nama": ["Andi", "Budi", "Citra", "Dewi", "Eka", "Fajar", "Gita"],
    "nilai": [85, 70, 78, 90, 88, 76, 95],
    "metode_belajar": [1, 2, 1, 3, 2, 3, 1]  # 1: Visual, 2: Auditori, 3: Kinestetik
}

# Membuat DataFrame
df = pd.DataFrame(students)

# Menggabungkan data nilai dan metode belajar untuk klustering
X = df[["nilai", "metode_belajar"]].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah kluster
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Melakukan klustering
kmeans.fit(X_scaled)

# Menambahkan hasil kluster ke DataFrame
df["kluster"] = kmeans.labels_

# Mapping metode belajar untuk keterbacaan
metode_mapping = {1: "Visual", 2: "Auditori", 3: "Kinestetik"}
df["metode_belajar"] = df["metode_belajar"].map(metode_mapping)

# Menampilkan hasil
print("Centroid kluster:")
print(kmeans.cluster_centers_)
print("\nData mahasiswa dengan kluster:")
print(df)

# Visualisasi hasil
plt.figure(figsize=(8, 6))
colors = ["red", "blue", "green"]

for cluster in range(num_clusters):
    cluster_points = df[df["kluster"] == cluster]
    plt.scatter(
        cluster_points["nilai"],
        cluster_points.index,  # Menggunakan index untuk plot kategori
        label=f"Kluster {cluster + 1}",
        color=colors[cluster]
    )

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200, c="yellow", marker="X", label="Centroid"
)

plt.title("Hasil Klustering Mahasiswa")
plt.xlabel("Nilai")
plt.ylabel("Metode Belajar")
plt.legend()
plt.grid(True)
plt.show()

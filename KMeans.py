import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Data contoh
data = {
    'nama': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'ipk': [3.8, 2.5, 3.0, 2.8, 3.5],
    'metode_belajar': ['visual', 'audio', 'kinetik', 'visual', 'audio']
}

# Membuat DataFrame
df = pd.DataFrame(data)

# One-Hot Encoding untuk metode_belajar
encoder = OneHotEncoder()
encoded_metode = encoder.fit_transform(df[['metode_belajar']]).toarray()
encoded_columns = encoder.get_feature_names_out(['metode_belajar'])

# Menggabungkan hasil encoding ke DataFrame utama
encoded_df = pd.DataFrame(encoded_metode, columns=encoded_columns)
df = pd.concat([df, encoded_df], axis=1)

# Menghapus kolom metode_belajar asli
df.drop(columns=['metode_belajar'], inplace=True)

# Menyiapkan data untuk clustering
X = df[['ipk'] + list(encoded_columns)]

# Menentukan jumlah kluster
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Menampilkan hasil clustering
print("Hasil Clustering:")
print(df)

# Visualisasi hasil clustering (berbasis IPK saja)
plt.scatter(df['ipk'], np.zeros_like(df['ipk']), c=df['cluster'], cmap='viridis', s=100)
plt.xlabel('IPK')
plt.title('Klustering Mahasiswa Berdasarkan IPK dan Metode Belajar')
plt.show()

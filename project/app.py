import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "File CSV tidak ditemukan."})
        
        file = request.files['file']
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Format file harus berupa CSV."})
        
        df = pd.read_csv(file, sep=';')
        
        if df.iloc[0].str.contains('Nama').any():
            df.columns = df.iloc[0]  # Set baris pertama sebagai header
            df = df[1:]  # Hapus baris header dari data
        
        df.columns = df.columns.str.strip()

        expected_columns = ['Nama', 'Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3','Mata_Pelajaran_4', 'Mata_Pelajaran_5', 'Visual', 'Auditori', 'Kinestetik']
        if not all(col in df.columns for col in expected_columns):
            return jsonify({"error": f"File CSV harus memiliki kolom: {', '.join(expected_columns)}."})
        
        valid_learning_styles = [0, 1]
        for col in ['Visual', 'Auditori', 'Kinestetik']:
            if not df[col].astype(int).isin(valid_learning_styles).all():
                return jsonify({"error": f"Kolom {col} hanya boleh berisi nilai 0 atau 1."})
        
        clustering_columns = ['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                              'Mata_Pelajaran_4', 'Mata_Pelajaran_5', 'Visual', 'Auditori', 'Kinestetik']
        for col in clustering_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=clustering_columns)
        
        df['Nilai_Rata_Rata'] = df[['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                                     'Mata_Pelajaran_4', 'Mata_Pelajaran_5']].mean(axis=1)

        bins = [0, 60, 80, 100]
        labels = ['Rendah', 'Sedang', 'Tinggi']
        df['Performa'] = pd.cut(df['Nilai_Rata_Rata'], bins=bins, labels=labels)

        scaler = StandardScaler()
        X = df[['Nilai_Rata_Rata', 'Visual', 'Auditori', 'Kinestetik']]
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, df['Cluster'])

        # PCA untuk mereduksi dimensi data menjadi 2D
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(X_scaled)
        
        # Visualisasi Clustering dengan KMeans dan PCA
        plt.figure(figsize=(10, 6))

        # Menambahkan plot KMeans dengan warna berbeda untuk setiap cluster
        sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='viridis', s=100, marker='o')

        # Menambahkan centroids ke dalam plot (pusat cluster)
        centroids = kmeans.cluster_centers_
        centroids_pca = pca.transform(centroids)  # Mengubah pusat cluster ke dalam komponen PCA
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', label='Centroids')

        plt.title('Visualisasi Clustering Berdasarkan Nilai dan Metode Belajar (KMeans)')
        plt.xlabel('PCA Komponen 1')
        plt.ylabel('PCA Komponen 2')
        plt.legend()

        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
        plt.close()

        # Menghitung kesimpulan tentang kelompok mahasiswa dan metode belajar yang dominan
        cluster_summary = {}
        for cluster_num in range(3):
            cluster_data = df[df['Cluster'] == cluster_num]
            learning_method_counts = cluster_data[['Visual', 'Auditori', 'Kinestetik']].sum(axis=0)
            dominant_method = learning_method_counts.idxmax()
            cluster_summary[cluster_num] = {
                "jumlah_mahasiswa": str(len(cluster_data)),
                "metode_belajar_dominan": dominant_method,
                "jumlah_metode_belajar": learning_method_counts.to_dict(),
                "performa_rata_rata": cluster_data['Nilai_Rata_Rata'].mean()
            }

        result = df.to_dict(orient='records')
        return jsonify({
            "result": result, 
            "silhouette_score": silhouette, 
            "plot_url": f"data:image/png;base64,{img_base64}", 
            "cluster_summary": cluster_summary
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
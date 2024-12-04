import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

        expected_columns = ['Nama', 'Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3',
                            'Mata_Pelajaran_4', 'Mata_Pelajaran_5', 'Visual', 'Auditori', 'Kinestetik']
        if not all(col in df.columns for col in expected_columns):
            return jsonify({"error": f"File CSV harus memiliki kolom: {', '.join(expected_columns)}."})
        
        clustering_columns = ['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                              'Mata_Pelajaran_4', 'Mata_Pelajaran_5']
        for col in clustering_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=clustering_columns)
        
        # Hitung nilai rata-rata
        df['Nilai_Rata_Rata'] = df[['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                                     'Mata_Pelajaran_4', 'Mata_Pelajaran_5']].mean(axis=1)

        # Pastikan kolom metode belajar diubah menjadi numerik (0/1)
        for col in ['Visual', 'Auditori', 'Kinestetik']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Lakukan clustering hanya berdasarkan Nilai_Rata_Rata
        scaler = StandardScaler()
        X = df[['Nilai_Rata_Rata']]
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, df['Cluster'])

        # Tentukan performa berdasarkan cluster
        cluster_means = df.groupby('Cluster')['Nilai_Rata_Rata'].mean().sort_values()
        cluster_performance = {cluster: label for cluster, label in zip(cluster_means.index, ['Rendah', 'Sedang', 'Tinggi'])}
        df['Performa'] = df['Cluster'].map(cluster_performance)

        # Menghitung kesimpulan tentang kelompok mahasiswa dan metode belajar yang dominan
        cluster_summary = {}
        for cluster_num in range(3):
            cluster_data = df[df['Cluster'] == cluster_num]
            
            # Perhitungan jumlah metode belajar berdasarkan kolom Visual, Auditori, Kinestetik
            learning_method_counts = cluster_data[['Visual', 'Auditori', 'Kinestetik']].sum(axis=0)
            
            # Validasi: Pastikan total metode belajar tidak melebihi jumlah mahasiswa
            assert learning_method_counts.sum() <= len(cluster_data), "Jumlah metode belajar melebihi jumlah mahasiswa."

            # Mencari metode belajar yang paling dominan
            dominant_method = learning_method_counts.idxmax()
            
            # Menyimpan hasil dalam format yang lebih jelas
            cluster_summary[cluster_num] = {
                "jumlah_mahasiswa": len(cluster_data),
                "metode_belajar_dominan": dominant_method,
                "jumlah_metode_belajar": learning_method_counts.to_dict(),
                "performa_rata_rata": cluster_data['Nilai_Rata_Rata'].mean()
            }

        # Visualisasi Clustering dengan Centroid
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_scaled[:, 0], y=[0] * len(X_scaled), hue=df['Cluster'], palette='viridis', s=100, marker='o')

        # Menambahkan titik centroid
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], [0] * len(centroids), s=200, c='red', marker='X', label='Centroid')
        
        plt.title('Clustering Berdasarkan Nilai Rata-Rata')
        plt.xlabel('Nilai Rata-Rata (Standardized)')
        plt.legend()

        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')
        plt.close()

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

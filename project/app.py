from flask import Flask, render_template, request, jsonify  # Mengimpor modul Flask dan fungsi terkait untuk membuat aplikasi web
import pandas as pd  # Mengimpor pandas untuk manipulasi data
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk visualisasi data
import seaborn as sns  # Mengimpor seaborn untuk visualisasi yang lebih baik
from sklearn.cluster import KMeans  # Mengimpor KMeans untuk algoritma clustering
from sklearn.preprocessing import StandardScaler  # Mengimpor StandardScaler untuk normalisasi data
from sklearn.metrics import silhouette_score  # Mengimpor silhouette_score untuk evaluasi clustering
from io import BytesIO  # Mengimpor BytesIO untuk menyimpan gambar dalam memori
import base64  # Mengimpor base64 untuk encoding gambar

app = Flask(__name__)  # Membuat instance aplikasi Flask

@app.route('/')  # Mendefinisikan route untuk halaman utama
def index():
    return render_template('index.html')  # Mengembalikan template HTML untuk halaman utama

@app.route('/process', methods=['POST'])  # Mendefinisikan route untuk memproses data yang diunggah
def process():
    try:
        if 'file' not in request.files:  # Memeriksa apakah file diunggah
            return jsonify({"error": "File CSV tidak ditemukan."})  # Mengembalikan pesan error jika tidak ada file
        
        file = request.files['file']  # Mengambil file dari request
        
        if not file.filename.endswith('.csv'):  # Memeriksa apakah file berformat CSV
            return jsonify({"error": "Format file harus berupa CSV."})  # Mengembalikan pesan error jika format salah
        
        df = pd.read_csv(file, sep=';')  # Membaca file CSV menjadi DataFrame
        
        if df.iloc[0].str.contains('Nama').any():  # Memeriksa apakah baris pertama mengandung 'Nama'
            df.columns = df.iloc[0]  # Mengatur baris pertama sebagai header
            df = df[1:]  # Menghapus baris header dari data
        
        df.columns = df.columns.str.strip()  # Menghapus spasi di sekitar nama kolom

        expected_columns = ['Nama', 'Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3',
                            'Mata_Pelajaran_4', 'Mata_Pelajaran_5', 'Visual', 'Auditori', 'Kinestetik']  # Mendefinisikan kolom yang diharapkan
        if not all(col in df.columns for col in expected_columns):  # Memeriksa apakah semua kolom yang diharapkan ada
            return jsonify({"error": f"File CSV harus memiliki kolom: {', '.join(expected_columns)}."})  # Mengembalikan pesan error jika kolom tidak lengkap
        
        clustering_columns = ['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                              'Mata_Pelajaran_4', 'Mata_Pelajaran_5']  # Mendefinisikan kolom untuk clustering
        for col in clustering_columns:  # Mengonversi kolom menjadi numerik
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Mengonversi kolom menjadi numerik, mengganti yang tidak valid dengan NaN
        
        df = df.dropna(subset=clustering_columns)  # Menghapus baris yang memiliki NaN di kolom clustering
        
        # Hitung nilai rata-rata
        df['Nilai_Rata_Rata'] = df[['Mata_Pelajaran_1', 'Mata_Pelajaran_2', 'Mata_Pelajaran_3', 
                                     'Mata_Pelajaran_4', 'Mata_Pelajaran_5']].mean(axis=1)  # Menghitung rata-rata nilai

        # Pastikan kolom metode belajar diubah menjadi numerik (0/1)
        for col in ['Visual', 'Auditori', 'Kinestetik']:  # Mengonversi kolom metode belajar menjadi numerik
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)  # Mengonversi dan mengganti NaN dengan 0

        # Visualisasi Data Sebelum Clustering
        plt.figure(figsize=(10, 6))  # Mengatur ukuran gambar
        sns.scatterplot(x=df.index, y=df['Nilai_Rata_Rata'], palette='viridis', s=100, marker='o')  # Membuat scatter plot
        plt.title('Data Sebelum Clustering')  # Menambahkan judul
        plt.xlabel('Indeks Mahasiswa')  # Menambahkan label sumbu x
        plt.ylabel('Nilai Rata-Rata')  # Menambahkan label sumbu y
        plt.legend()  # Menampilkan legenda

        pre_cluster_img_stream = BytesIO()  # Membuat stream untuk menyimpan gambar
        plt.savefig(pre_cluster_img_stream, format='png')  # Menyimpan gambar ke stream
        pre_cluster_img_stream.seek(0)  # Mengatur posisi stream ke awal
        pre_cluster_img_base64 = base64.b64encode(pre_cluster_img_stream.getvalue()).decode('utf-8')  # Mengonversi gambar ke base64
        plt.close()  # Menutup gambar

        # Lakukan clustering hanya berdasarkan Nilai_Rata_Rata
        scaler = StandardScaler()  # Membuat instance StandardScaler
        X = df[['Nilai_Rata_Rata']]  # Mengambil kolom nilai rata-rata untuk clustering
        X_scaled = scaler.fit_transform(X)  # Melakukan normalisasi pada data
        
        kmeans = KMeans(n_clusters=3, random_state=42)  # Membuat instance KMeans dengan 3 cluster
        df['Cluster'] = kmeans.fit_predict(X_scaled)  # Melakukan clustering dan menambahkan hasil ke DataFrame
        
        silhouette = silhouette_score(X_scaled, df['Cluster'])  # Menghitung nilai silhouette untuk evaluasi clustering

        # Tentukan performa berdasarkan cluster
        cluster_means = df.groupby('Cluster')['Nilai_Rata_Rata'].mean().sort_values()  # Menghitung rata-rata nilai per cluster
        cluster_performance = {cluster: label for cluster, label in zip(cluster_means.index, ['Rendah', 'Sedang', 'Tinggi'])}  # Menentukan label performa
        df['Performa'] = df['Cluster'].map(cluster_performance)  # Menambahkan kolom performa ke DataFrame

        # Menghitung kesimpulan tentang kelompok mahasiswa dan metode belajar yang dominan
        cluster_summary = {}  # Inisialisasi dictionary untuk ringkasan cluster
        for cluster_num in range(3):  # Iterasi untuk setiap cluster
            cluster_data = df[df['Cluster'] == cluster_num]  # Mengambil data untuk cluster tertentu
            
            # Perhitungan jumlah metode belajar berdasarkan kolom Visual, Auditori, Kinestetik
            learning_method_counts = cluster_data[['Visual', 'Auditori', 'Kinestetik']].sum(axis=0)  # Menghitung jumlah metode belajar
            
            # Validasi: Pastikan total metode belajar tidak melebihi jumlah mahasiswa
            assert learning_method_counts.sum() <= len(cluster_data), "Jumlah metode belajar melebihi jumlah mahasiswa."  # Memastikan validitas data

            # Mencari metode belajar yang paling dominan
            dominant_method = learning_method_counts.idxmax()  # Menentukan metode belajar yang paling banyak
            
            # Menyimpan hasil dalam format yang lebih jelas
            cluster_summary[cluster_num] = {
                "jumlah_mahasiswa": len(cluster_data),  # Menyimpan jumlah mahasiswa di cluster
                "metode_belajar_dominan": dominant_method,  # Menyimpan metode belajar dominan
                "jumlah_metode_belajar": learning_method_counts.to_dict(),  # Menyimpan jumlah metode belajar
                "performa_rata_rata": cluster_data['Nilai_Rata_Rata'].mean()  # Menyimpan rata-rata performa
            }

        # Visualisasi Clustering dengan Centroid
        plt.figure(figsize=(10, 6))  # Mengatur ukuran gambar
        sns.scatterplot(x=X_scaled[:, 0], y=[0] * len(X_scaled), hue=df['Cluster'], palette='viridis', s=100, marker='o')  # Membuat scatter plot untuk clustering

        # Menambahkan titik centroid
        centroids = kmeans.cluster_centers_  # Mengambil titik centroid dari clustering
        plt.scatter(centroids[:, 0], [0] * len(centroids), s=200, c='red', marker='X', label='Centroid')  # Menambahkan centroid ke plot
        
        plt.title('Clustering Berdasarkan Nilai Rata-Rata')  # Menambahkan judul
        plt.xlabel('Nilai Rata-Rata (Standardized)')  # Menambahkan label sumbu x
        plt.legend()  # Menampilkan legenda

        img_stream = BytesIO()  # Membuat stream untuk menyimpan gambar
        plt.savefig(img_stream, format='png')  # Menyimpan gambar ke stream
        img_stream.seek(0)  # Mengatur posisi stream ke awal
        img_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')  # Mengonversi gambar ke base64
        plt.close()  # Menutup gambar

        result = df.to_dict(orient='records')  # Mengonversi DataFrame menjadi dictionary
        return jsonify({  # Mengembalikan hasil dalam format JSON
            "result": result, 
            "silhouette_score": silhouette, 
            "plot_url": f"data:image/png;base64,{img_base64}", 
            "pre_cluster_plot_url": f"data:image/png;base64,{pre_cluster_img_base64}",
            "cluster_summary": cluster_summary
        })
    
    except Exception as e:  # Menangani exception
        return jsonify({"error": str(e)})  # Mengembalikan pesan error

if __name__ == '__main__':  # Memeriksa apakah file ini dijalankan sebagai program utama
    app.run(debug=True)  # Menjalankan aplikasi Flask dalam mode debug
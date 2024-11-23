from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Ambil data dari form
    try:
        jumlah_siswa = int(request.form['rows'])
        jumlah_mata_pelajaran = int(request.form['columns'])
        
        # Ambil nilai dan metode belajar dari form
        data_nilai = []
        for i in range(jumlah_siswa):
            nilai = request.form.getlist(f'nilai_{i+1}')
            metode = request.form.getlist(f'metode_{i+1}')
            data_nilai.append(list(map(float, nilai)) + list(map(float, metode)))
        
        # Kolom data (dinamis sesuai input)
        columns = [f'Mata_Pelajaran_{i+1}' for i in range(jumlah_mata_pelajaran)] + ['Visual', 'Auditori', 'Kinestetik']
        df = pd.DataFrame(data_nilai, columns=columns)

        # Proses Klustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Evaluasi Klustering
        silhouette = silhouette_score(X_scaled, df['Cluster'])

        # Format hasil untuk frontend
        result = df.to_dict(orient='records')
        return jsonify({"result": result, "silhouette_score": silhouette})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

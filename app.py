from flask import Flask, request, jsonify
import pandas as pd
import os
import kmeans_with_library
import db_operations
import logging

app = Flask(__name__)

# Loglama yapılandırması
db_operations.setup_logging()

# Veri yolu ve tablo adı oluştur
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dosya yükleme yolu
@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

   
    # Dosya kaydedilmesi
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # CSV dosyasını pandas DataFrame'e yükle
    df = pd.read_csv(file_path)

    # Tablo ismini belirle
    table_name = os.path.splitext(file.filename)[0]

    # Veritabanı bağlantısını kur
    conn = db_operations.connect_db()
    
    if conn is not None:
        # Tabloyu oluştur
        db_operations.create_table(table_name, conn, df)

        # Verileri veritabanına ekle
        db_operations.insert_data(table_name, df)

        # KMeans işlemlerini çalıştır
        kmeans_with_library.main(table_name)

        return jsonify({"message": "File processed and KMeans executed successfully"}), 200
    else:
        logging.error("Veritabanına bağlanılamadı.")
        return jsonify({"error": "Failed to connect to the database"}), 500
  

if __name__ == "__main__":
    app.run(debug=True)

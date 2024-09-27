import kmeans_with_library
import pandas as pd
import os
import db_operations
import logging

"""
Bu kod, bir CSV dosyasından verileri okuyarak bir PostgreSQL veritabanında tablo oluşturur, verileri ekler ve KMeans algoritmasını çalıştırır. 

1. Gerekli modüller (kmeans_with_library, pandas, os, db_operations, logging) içe aktarılır.
2. Okunacak CSV dosyasının yolu 'data_path' değişkeninde tanımlanır.
3. main() fonksiyonu, kodun yürütüldüğü ana bölümdür:
   - Loglama başlatılır.
   - CSV dosyası pandas DataFrame'e yüklenir.
   - CSV dosyasının adı, veritabanındaki tablo adı olarak belirlenir.
   - Veritabanına bağlantı sağlanır.
   - Bağlantı başarılıysa:
     - Veritabanında uygun bir tablo oluşturulur.
     - DataFrame'deki veriler bu tabloya eklenir.
     - KMeans algoritması çalıştırılır.
   - Bağlantı başarısız olursa hata günlüğüne kaydedilir.
"""

# Veri yolunu tanımla
data_path = "data/deneme_.csv"

def main():

    logging.info("Kod çalıştırılıyor.")

    db_operations.setup_logging()

    # Veriyi oku
    df = pd.read_csv(data_path)
    
    # Tablo ismini belirle
    table_name = os.path.splitext(os.path.basename(data_path))[0]

    # Veritabanı bağlantısını kur
    conn = db_operations.connect_db()
    
    if conn is not None:
        # Tabloyu oluştur
        db_operations.create_table(table_name, conn, df)

        # Verileri veritabanına ekle
        db_operations.insert_data(table_name, df)

        # KMeans işlemlerini çalıştır
        kmeans_with_library.main(table_name)
    else:
        logging.error("Veritabanına bağlanılamadı.")

if __name__ == "__main__":
    main()

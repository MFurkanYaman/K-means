import warnings
import time
import config
import logging
import pandas as pd
import os
import app
import db_operations
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
       

def normalize_data(df):

    """
    Bu fonksiyon, verilen veri setindeki değerleri MinMaxScaler kullanarak 0 ile 1 arasına ölçeklendirir. 

    Parametreler:
    df: Normalize edilecek veri seti. Veri çerçevesi, sayısal sütunlar içermelidir.

    Return Değeri:
    scaler_fit: Normalize edilmiş veri seti. 0 ile 1 arasında değerler içeren bir numpy dizisi döndürür. 
    """
    
    try:
        scaler = MinMaxScaler()
        scaler_fit=scaler.fit_transform(df)
        return scaler_fit 
    except Exception as e:
        logging.error(e)

def find_best_value_kmeans(scaled_data, cluster):

    """
    KMeans algoritması kullanarak en iyi küme sayısını bulur.

    Parametreler:
    scaled_data (ndarray veya DataFrame): KMeans algoritması için ölçeklendirilmiş veri seti.
    cluster (int): Denenecek maksimum küme sayısı.

    Return Değerleri:
    best_k: En iyi küme sayısı.
    best_value: En yüksek Silhouette skoru.
    best_model: En başarılı KMeans modeli.

    Bu fonksiyon, verilen ölçeklendirilmiş veri seti üzerinde KMeans algoritmasını 
    uygular ve en yüksek Silhouette skoruna sahip küme sayısını belirler. 
    """

    try:

        best_value = -1
        best_k = 1
        for k in range(2, cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            value = silhouette_score(scaled_data, labels)
            if value > best_value:
                best_value = value
                best_k = k
                best_model = kmeans
        
        return  best_k,best_value,best_model
    
    except Exception as e:
        logging.error(e)

def save_results(df,execution_time,best_k,score):

    """
    Sonuçları CSV dosyasına kaydeder.

    Parametreler:
    df: KMeans algoritması ile elde edilen verileri içeren DataFrame.
    execution_time: Algoritmanın çalışma süresi (saniye cinsinden).
    best_k: En iyi küme sayısı.
    score: KMeans algoritması için hesaplanan Silhouette skoru.

    Bu fonksiyon, verilen DataFrame ve performans metriklerini kullanarak 
    'results/result.csv' dosyasına sonuçları kaydeder. Çalışma süresi ve Silhouette skoru 
    gibi performans metriklerinin yalnızca ilk altı hanesi kaydedilir.
    """    
 
    try:

        results_summary = pd.DataFrame({
                'Execution Time (s)': [str(execution_time)[:6]],  
                'Best K': [best_k],
                'Silhouette Score': [str(score)[:6]]  
            })        
        df.to_csv("results/result.csv", index=False)
        results_summary.to_csv("results/result.csv",mode="a",index=False)

    except Exception as e:
        logging(e)
        

def main(table_name):
    """
    Kod akışını yöneten fonksiyon.
    Sırasıyla İşlevi:
    - Veritabanına bağlanır.
    - Verileri alır.
    - Verileri normalize eder.
    - KMeans algoritmasını kullanarak en iyi küme sayısını bulur.
    - Sonuçları bir CSV dosyasına kaydeder.
    - İşlem süresini kaydeder.
    
    Hata durumunda, hata kaydı oluşturur.
    """
    try:
        start_time = time.time()

      

        # Veritabanı bağlantısı
        conn = db_operations.connect_db()
        
        # db_operations dosyasından verileri dataframe'e çevir.
        df = db_operations.get_data_from_db(conn, table_name)

        # Normalizasyon
        if df is not None:
            scaled_data = normalize_data(df)
        
            # En uygun küme sayısını bul
            max_cluster_num = 8
            best_k, score, best_model_kmeans = find_best_value_kmeans(scaled_data, max_cluster_num)
            df["output"] = best_model_kmeans.labels_

            # Veritabanını kapat
            conn.close()

            end_time = time.time()
            execution_time = end_time - start_time
            print(df)

            # Sonuçları yazdır.
            save_results(df, execution_time, best_k, score)
        else:
            logging.error(f"{table_name} tablosundan veri alınamadı.")
            
    except Exception as e:
        logging.error(e)

if __name__ == "__main__":  
    main()

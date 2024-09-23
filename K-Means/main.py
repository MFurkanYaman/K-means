import pandas as pd
import psycopg2
import warnings
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


warnings.filterwarnings("ignore")

def connect_db(dbname, host, user, password, port,timestamp):
    try:
        return psycopg2.connect(database=dbname, host=host, user=user, password=password, port=port)
    except Exception as e:
        file=open("log1.txt","a")
        file.write(f"{timestamp} - {e}\n")
        file.close()

def data_from_db(conn, query,timestamp):
    """Verilen SQL sorgusuna göre veritabanından verileri çeker ve bir pandas DataFrame'ine dönüştürür."""
    try:
        return pd.read_sql_query(query, conn)
    
    except Exception as e:
        file=open("log1.txt","a")
        file.write(f"{timestamp} - {e}.\n")
        file.close()

def normalize_data(df):
    """Verilen veriyi MinMaxScaler kullanarak normalize eder."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def find_best_value_kmeans(scaled_data, cluster):
    """
    KMeans algoritması ve silhouette skor kullanarak en iyi küme sayısını bulur ve modeli döndürür.

    scaled_data: KMeans algoritması için ölçeklendirilmiş veri seti.
    cluster: Denenecek maksimum küme sayısı (kümelerin sayısı).
    best_model: En yüksek Silhouette skoruna sahip KMeans modeli.

    """

    best_value = -1
    best_k = 1
    for k in range(2, cluster + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        value = silhouette_score(scaled_data, labels)
        print(f"Euclidean Distance için K={k} değeri, Silhouette Score: {value}")
        if value > best_value:
            best_value = value
            best_k = k
            best_model = kmeans
    print(f"En iyi küme sayısı: {best_k}, Silhouette Skoru: {best_value}")
    return  best_model

def execute_log(timestamp):
    """
    Çalıştırma zamanı ile bir log dosyasına işlem kaydı ekler.
    """
    file=open("log1.txt","a")
    file.write(f"{timestamp} - Code executed.\n")
    file.close()


def main():

    get_time=datetime.datetime.now()
    timestamp = get_time.strftime("%d.%m.%Y %H:%M:%S")

    #Log kayıt
    execute_log(timestamp)

    # Veritabanı bağlantısı
    conn = connect_db("dbKmeans", "localhost", "postgres", "1234", "5432",timestamp)
    


    # Verileri al
    query = "SELECT * FROM data;"  
    df = data_from_db(conn, query,timestamp)
    
    print("Veri Seti: ")
    print(df)
    
    # Normalizasyon
    scaled_data = normalize_data(df)
    
    # Öklid için en uygun küme sayısını bul
    max_cluster_num = 2
    best_model_kmeans = find_best_value_kmeans(scaled_data, max_cluster_num)
    df["output"] = best_model_kmeans.labels_
    
    
    # Sonuçları yazdır
    
    print(df)
    
    # Veritabanını kapat
    conn.close()

if __name__ == "__main__":
    main()

import psycopg2
import warnings
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

def connect_db(dbname, host, user, password, port):
    return psycopg2.connect(database=dbname, host=host, user=user, password=password, port=port)

def data_from_db(conn, query):
    return pd.read_sql_query(query, conn)

def normalize_data(df):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def find_best_value_kmeans(scaled_data, cluster):
    best_value = -1
    distance_metric= ["euclidean", "manhattan", "cosine"]

    for metric in distance_metric:
        for k in range(2, cluster + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            value = silhouette_score(scaled_data, labels, metric=metric)
            #print(f"Metrik: {metric}, k: {k}, Silhouette Skoru: {value}")

            if value > best_value:
                best_value = value
                best_k = k
                best_model = kmeans
                best_metric = metric

    print(f"En iyi küme sayısı: {best_k}, Silhouette Skoru: {best_value}, Mesafe Metrik: {best_metric}")
    return  best_model
    
def main():
    # Veritabanı bağlantısı
    conn = connect_db("dbKmeans", "localhost", "postgres", "1234", "5432")
    
    # Verileri al
    query = "SELECT * FROM data;"  
    df = data_from_db(conn, query)
    
    print("Veri Seti: ")
    print(df)
    
    # Normalizasyon
    scaled_data = normalize_data(df)
    
    # Öklid için en uygun küme 
    max_cluster_num = 15
    best_model_kmeans = find_best_value_kmeans(scaled_data, max_cluster_num)
    df["output"] = best_model_kmeans.labels_
    
    
    # Sonuçları yazdır
    
    print(df)
    
    # Veritabanını kapat
    conn.close()

if __name__ == "__main__":
    main()

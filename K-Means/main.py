import psycopg2
import warnings
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids  # KMedoids ekleniyor
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
    fastest_time = float('inf') #sonsuz sayı
    algorithms=["lloyd","elkan"]
   
    for algorithm in algorithms:
        
        for k in range(2, cluster + 1):
            start_time=time.time()

            kmeans = KMeans(n_clusters=k, random_state=42, algorithm=algorithm)
            labels = kmeans.fit_predict(scaled_data)
            value = silhouette_score(scaled_data, labels)

            end_time=time.time()

            total_time=end_time-start_time            

            if value > best_value:
                best_value = value
                best_k = k
                best_model = kmeans

            if total_time < fastest_time:
                fastest_time = total_time
                fastest_k = k
                fastest_value = value
                alg_name=algorithm
            
                
    print(f"En iyi küme sayısı: {best_k}, Silhouette Skoru: {best_value}, Çalışma Süresi: {time_dict[best_k]}")
    print(f"En hızlı çalışma süresine sahip algoritma {alg_name}, küme sayısı: {fastest_k}, Silhouette Skoru: {fastest_value}, Çalışma Süresi: {fastest_time}")
    
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
    
    # Öklid için en uygun küme sayısını bul
    max_cluster_num = 10
    best_model_kmeans = find_best_value_kmeans(scaled_data, max_cluster_num)
    df["output_euclidean"] = best_model_kmeans.labels_
    
    
    # Sonuçları yazdır
    
    print(df)
    
    # Veritabanını kapat
    conn.close()

if __name__ == "__main__":
    main()

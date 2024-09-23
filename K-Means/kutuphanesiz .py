import psycopg2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy
import warnings

warnings.filterwarnings("ignore")


def connect_db(dbname, host, user, password, port):
    """Veritabanına bağlantı gerçekleştirir."""
    return psycopg2.connect(database=dbname, host=host, user=user, password=password, port=port)

def data_from_db(conn, query):
    """Verilen SQL sorgusuna göre veritabanından verileri çeker ve bir pandas DataFrame'ine dönüştürür."""
    return pd.read_sql_query(query, conn)

def normalize_data(df):
    """Verilen veriyi MinMaxScaler kullanarak normalize eder."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def calculate_distance(data, centroids, centroids_num):
    """Veri noktaları ile merkezler arasındaki uzaklığı Öklid yöntemiyle hesaplar ve sonuçları döndürür."""
    distance_dict = {}
    for i in range(centroids_num):
        distances = []
        for j in range(len(data)):
            distance = np.sqrt(np.sum((data[j] - centroids[i])**2))
            distances.append(distance)
        distance_dict[i] = distances
    return distance_dict


def find_clusters(distance_dict, row_num, centroids_num):

    """
    Her veri noktasını, merkezlerle olan mesafelerine göre en yakın kümeye atar.

    Parametreler:

    distance_dict: Her bir merkez için veri noktalarına olan mesafeleri içeren sözlük.
    row_num: Veri setindeki satır sayısı (veri noktalarının sayısı).
    centroids_num: Merkezlerin sayısı (kümelerin sayısı).
    clusters: Sözlük olarak her bir merkez için, o merkeze atanmış veri noktalarının
              indekslerini içeren kümeler  döndürür.
    """

    clusters = {}
    for i in range(row_num):
        min_value = float("inf")
        for j in range(centroids_num):
            if distance_dict[j][i] < min_value:
                min_value = distance_dict[j][i]
                min_key = j
        if min_key not in clusters:
            clusters[min_key] = []
        clusters[min_key].append(i)
    return clusters

def update_centroids(data, clusters, centroids_num):
    """
    Her bir kümedeki veri noktalarına göre merkezlerin(Centroids) konumunu günceller.

    Parametreler:
    
    clusters: Kümeleri temsil eden, sözlük.
    centroids_num: Merkezlerin sayısı (kümelerin sayısı).
    new_centroids: Güncellenmiş merkezleri içerir.
    """
    new_centroids = []
    for i in range(centroids_num):
        
        total = np.zeros(data.shape[1], dtype=np.float64)
        for inside_value in clusters[i]:
            total += data[inside_value]
        new_centroid = total / len(clusters[i])
        new_centroids.append(new_centroid)
        
    return np.array(new_centroids)

def kmeans_func(data, centroids, centroids_num):
    row_num = data.shape[0]
    old_clusters={}

    for i in range(centroids_num):
        old_clusters[i]= []

    iterations = 0

    while True:
        distance_dict = calculate_distance(data, centroids, centroids_num)
        clusters = find_clusters(distance_dict, row_num, centroids_num)
        centroids = update_centroids(data, clusters, centroids_num)
        
        if clusters== old_clusters:
            break
        else:
            old_clusters = copy.deepcopy(clusters)

        iterations += 1

    print(f"Döngü sayısı: {iterations}")
    print(f"Son centroidler: {centroids}")
    return centroids, clusters

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
    
    
    centroids_num = 2 

    #Scaled data içerisinden seçilen rastgeel centroids_num kadar veri noktası centroids olur.
    random_indices = np.random.choice(scaled_data.shape[0], centroids_num, replace=False)
    first_centroids = scaled_data[random_indices, :]

    print(first_centroids)

    # K-Means 
    final_centroids, final_clusters = kmeans_func(scaled_data, first_centroids, centroids_num)
    
    # print(f"Kümeler: {final_clusters}")
    # print(f"Final Merkez Noktaları: {final_centroids}")

    df["output"]=-1
   
    print(final_clusters.items())
    for cluster_label, index in final_clusters.items():
        
        # print("a",index)
        # print("b",cluster_label)
        
        df.loc[index, "output"] = cluster_label # indexte cluster labelsı atar


    print(df)


    
    
    # Veritabanını kapat
    conn.close()

if __name__ == "__main__":
    main()

import psycopg2
import time
import datetime
import copy
import pandas as pd
import warnings
import openpyxl
import numpy as np
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def connect_db(dbname, host, user, password, port,timestamp):
    """Veritabanına bağlantı gerçekleştirir."""
    try:
        return psycopg2.connect(database=dbname, host=host, user=user, password=password, port=port)
    
    except Exception as e:
        file=open("log_for_kmeans_without_library.txt","a")
        file.write(f"{timestamp} - {e}\n")
        file.close()
        

def data_from_db(conn, query,timestamp):
    """Verilen SQL sorgusuna göre veritabanından verileri çeker ve bir pandas DataFrame'ine dönüştürür."""
    try:
        return pd.read_sql_query(query, conn)
    
    except Exception as e:
        file=open("log_for_kmeans_without_library.txt","a")
        file.write(f"{timestamp} - {e}.\n")
        file.close()
        

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

    while True:
        distance_dict = calculate_distance(data, centroids, centroids_num)
        clusters = find_clusters(distance_dict, row_num, centroids_num)
        centroids = update_centroids(data, clusters, centroids_num)
        
        if clusters== old_clusters:
            break
        else:
            old_clusters = copy.deepcopy(clusters)

    return clusters


def execute_log(timestamp):
    """
    Code execute bilgilerini log dosyasına yazdırır.
    """
    file=open("log_for_kmeans_without_library.txt","a")
    file.write(f"{timestamp} - Code executed.\n")
    file.close()

def calculate_silhoutte_score(df, centroids_num):
    """
    Silhoutte Score hesaplaması yaparak en iyi küme sayısını bulmayı hedefler.
    a(i), bir küme içindeki noktaların birbirlerine olan ortalama mesafelerini, 
    b(i) ise bir noktanın en yakın diğer kümedeki noktaların ortalama mesafesini temsil eder. 
    Bu değerler kullanılarak her küme için Silhouette Score hesaplanır ve genel bir ortalama döndürülür.
    
    """
    scaled_data = pd.DataFrame(normalize_data(df.iloc[:, :3]), columns=df.columns[:3])
    scaled_data = pd.concat([scaled_data, df.iloc[:, -1]], axis=1)

    centroids_a_value = {}
    centroids_b_value = {}

    # a(i) değerlerini hesapla (aynı küme içindeki mesafeler)
    for i in range(centroids_num):
        total_a_mean = 0
        clusters = scaled_data[scaled_data["output"] == i].drop(columns=["output"])
        
        for j in range(len(clusters)):
            distances_a = []
            for k in range(len(clusters)):
                if j != k:  # Kendisi ile mesafeyi hesaplama
                    distance_a = np.sqrt(np.sum((clusters.iloc[j].values - clusters.iloc[k].values) ** 2))
                    distances_a.append(distance_a)
            a_mean = np.mean(distances_a)
            total_a_mean += a_mean
        total_a_mean /= len(clusters)
        centroids_a_value[i] = total_a_mean

    # b(i) değerlerini hesapla (diğer kümelerdeki mesafeler)
    for i in range(centroids_num):
        clusters = scaled_data[scaled_data["output"] == i].drop(columns=["output"])
        for j in range(len(clusters)):  
            min_b_mean = float("inf")  
            for k in range(centroids_num):  
                if i != k:  
                    other_clusters = scaled_data[scaled_data["output"] == k].drop(columns=["output"])
                    distances_b = []
                    for m in range(len(other_clusters)):
                        distance_b = np.sqrt(np.sum((clusters.iloc[j].values - other_clusters.iloc[m].values) ** 2))
                        distances_b.append(distance_b)
                    b_mean = np.mean(distances_b)  # O diğer kümedeki noktalarla olan ortalama mesafe
                    if b_mean < min_b_mean:  # En küçük ortalama mesafeyi al
                        min_b_mean = b_mean
            centroids_b_value[(i, j)] = min_b_mean # centroids_b_value = (küme_no,indeks):en kısa mesafe 
            
    #b yi kümelerine göre ayır
    same_group = {}
    for (cluster_label, index), distance in centroids_b_value.items():
        if cluster_label not in same_group:
            same_group[cluster_label] = []
        same_group[cluster_label].append(distance)
   
    #b ortalamasını ubl 
    for cluster_label, distances in same_group.items():
        if (len(distances)>0):
            mean_total_b = sum(distances) / len(distances)
            same_group[cluster_label]=mean_total_b
        else:
            same_group[cluster_label] = None
    
    score=[]
    for i in range(centroids_num):
        if np.isnan(centroids_a_value[i])  or np.isnan(same_group[i]):
            continue
        else:
            score.append((same_group[i]-centroids_a_value[i])/max(same_group[i],centroids_a_value[i]))
    score_mean=np.mean(score)       
    return score_mean 

def write_results_to_excel(execution_time, best_centroids_num, best_silhouette_score, best_df):
    """
    Kod çıktılarını Excel Dosyasına kaydeder.
    """
    
    results = pd.DataFrame({
        "Info": ["Execution Time", "Best Cluster Number", "Best Silhouette Score"],
        "Values": [f"{execution_time:.6f} seconds", best_centroids_num, best_silhouette_score]
    })

    
    with pd.ExcelWriter('kmeans_results.xlsx', engine='openpyxl') as writer:
        # Bilgiler hakkında yaz
        results.to_excel(writer, sheet_name='Results', index=False, startrow=0)
        
        # df çıktısını yaz
        best_df.to_excel(writer, sheet_name='Results', index=False, startrow=len(results) + 2)


def main():

    start_time = time.time()

    get_time=datetime.datetime.now()
    timestamp = get_time.strftime("%d.%m.%Y %H:%M:%S")
    execute_log(timestamp)

    # Veritabanı bağlantısı
    conn = connect_db("dbKmeans", "localhost", "postgres", "1234", "5432",timestamp)
    
    # Verileri al
    query = "SELECT * FROM data;"  
    df = data_from_db(conn, query,timestamp)
    
    # Normalizasyon
    scaled_data = normalize_data(df)
    
    score_dict={}
    df_dict={}
    max_centroids_num = 7

    # Küme döngüsü sağlar
    for centroids_num in range (2,max_centroids_num+1):

        #Scaled data içerisinden seçilen rastgele centroids_num kadar veri noktası centroids olur.
        random_indices = np.random.choice(scaled_data.shape[0], centroids_num, replace=False)
        first_centroids = scaled_data[random_indices, :]

        # K-Means 
        final_clusters = kmeans_func(scaled_data, first_centroids, centroids_num)

        temp_df = df.copy()
        temp_df["output"] = -1
                
        for cluster_label, index in final_clusters.items():
            temp_df.loc[index, "output"] = cluster_label # indexte cluster labelsı atar

        silhoutte_score=calculate_silhoutte_score(temp_df,centroids_num)
        score_dict[centroids_num]=silhoutte_score
        df_dict[centroids_num] = temp_df

    best_centroids_num = max(score_dict, key=score_dict.get)
    best_df = df_dict[best_centroids_num]
    
    # Veritabanını kapat
    conn.close()

    end_time = time.time()
    execution_time = end_time - start_time

    #Çıktı sonuçlarını excele yazdır.
    write_results_to_excel(execution_time, best_centroids_num, score_dict[best_centroids_num], best_df)

if __name__ == "__main__":
    main()

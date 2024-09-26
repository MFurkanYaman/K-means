import psycopg2
import warnings
import time
import config
import logging
import numpy as np
import pandas as pd
import db_operations
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

def connect_db():

    """
    Belirli ayarlarla (host, port, database, user, password) 
    bir PostgreSQL veritabanına bağlantı kurmaya çalışır. Bağlantı başarılı olursa, 
    bağlantı nesnesini döndürür. Aksi takdirde, hata log dosyasına kaydedilir.

    Return Değeri:
    connection: psycopg2 bağlantı nesnesi. 
                Başarılı bir bağlantı kurulduğunda döndürülür; 
                hata durumunda None döner.
    """
    
    try:
        connection = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        return connection
    except Exception as e:
        logging.error(e)


def data_from_db(conn, query):
    
    """
    Verilen SQL sorgusunu kullanarak veritabanından verileri çeker 
    ve sonuçları bir pandas DataFrame'ine dönüştürür.

    Parametreler:
    conn: Veritabanı bağlantısını temsil eden psycopg2 bağlantı nesnesi. 
    query: Veritabanına gönderilecek SQL sorgusu.

    Return Değeri:
    DataFrame: Tüm sütunları sayısal ve boş değer içermeyen SQL sorgusunun 
    sonucunu içeren bir pandas DataFrame nesnesi döndürür. Eğer veri setindeki 
    sütunlar sayısal değilse veya boş değer varsa geri dönüş yapılmaz ve hata kaydedilir.
    """


    try:
        dataframe = pd.read_sql_query(query, conn)
        numeric_columns = dataframe.select_dtypes(include=[np.number])
        empty_check=dataframe.isnull().any().any()
        if len(numeric_columns.columns) == len(dataframe.columns) and empty_check==False:
            return dataframe

    except Exception as e:
        logging.error(e)
       

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

def setup_logging(log_file=config.LOG_FILE,level=logging.INFO):

    """
    Bu fonksiyon, verilen log dosyası ve loglama seviyesine göre 
    loglama ayarlarını yapılandırır. Loglar, belirtilen dosyaya yazılır 
    ve zaman damgası, log seviyesi ve mesaj formatında kaydedilir

    Parametreler:
    log_file (str): Log dosyasının adı. Varsayılan olarak config.LOG_FILE'den alınır.
    level (int): Loglama seviyesini belirler. Varsayılan olarak logging.INFO seviyesidir.
    
    """

    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )

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
        

def main():

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

        #Log kayıtlarını oluştur.
        setup_logging()

        logging.info("Code executed")
        
        # Veritabanı bağlantısı
        conn = db_operations.connect_db()
    
        # Verileri al
        query = "SELECT * FROM data;"  
        df = data_from_db(conn, query)

        # Normalizasyon
        scaled_data = normalize_data(df)
        
        # En uygun küme sayısını bul
        max_cluster_num = 8
        best_k,score,best_model_kmeans = find_best_value_kmeans(scaled_data, max_cluster_num)
        df["output"] = best_model_kmeans.labels_

        # Veritabanını kapat
        conn.close()

        end_time = time.time()
        execution_time = end_time - start_time
        print(df)

        # Sonuçları yazdır.
        save_results(df,execution_time,best_k,score)

    except Exception as e:
        logging.error(e)

if __name__ == "__main__":  
    main()


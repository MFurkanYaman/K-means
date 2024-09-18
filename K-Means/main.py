import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def connect_db(dbname, host, user, password, port):
    return psycopg2.connect(database=dbname, host=host, user=user, password=password, port=port)


def data_from_db(conn, query):
    return pd.read_sql_query(query, conn)


def normalize_data(df):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)


def kmeans_fit(scaled_data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)
    return kmeans.labels_


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

    # Kullanıcıdan k sayısını al
    num_clusters = int(input("Küme sayısı giriniz: "))

    # K-means kümeleme
    df['output'] = kmeans_fit(scaled_data, num_clusters)

    # Sonuçları yazdır
    print("K-Means Veri Seti: ")
    print(df)

    # Veritabanını kapat
    conn.close()


if __name__ == "__main__":
    main()

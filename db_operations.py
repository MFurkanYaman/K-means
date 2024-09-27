import config
import app
import os
import psycopg2
import logging
import numpy as np
import pandas as pd

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
        logging.info("Veri tabanına bağlanıldı.")
        return connection

    except Exception as e:
       logging.error()
    
def create_table(table_name,conn,df):

    """
    Verilen bir CSV dosyasından veri alarak, bu verileri temsil edecek 
    bir PostgreSQL veritabanı tablosu oluşturur. Tablo adı, CSV dosyasının 
    adından türetilir. Fonksiyon, her bir sütunu 'TEXT' veri tipiyle tanımlayarak 
    SQL CREATE TABLE sorgusunu oluşturur ve veritabanında uygulamak için 
    bağlantıyı kullanır.

    Parametreler:
    data_path: CSV dosyasının dosya yolu. Tablo adı, bu dosya adından türetilir.
    conn: psycopg2 bağlantı nesnesi. Veritabanına bağlanmak için kullanılır.
    """
    try:
        
        columns = df.columns

        columns_definition = ""

        for col in columns:
            columns_definition += f"{col} INTEGER, "

        columns_definition = columns_definition.rstrip(", ")

        create_table_query = f"CREATE TABLE {table_name} ({columns_definition});"

        cur = conn.cursor()
        
        cur.execute(create_table_query)
        conn.commit()

        logging.info(f"{table_name} isimli tablo başarıyla oluşturuldu.")

    except Exception as e:
        logging.error(e)

def insert_data(table_name,df):

    conn = connect_db()
    cur = conn.cursor()
    columns = ", ".join(df.columns)

    try:
        for index, row in df.iterrows():
            values = ""
            for val in row:
                values += f"'{val}', "
            values = values.rstrip(", ")
            
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values});"
            cur.execute(insert_query)
            conn.commit()
            logging.info(f"Veriler {table_name} tablosuna başarıyla eklendi!")
    except Exception as e:
            logging.error(e)

def get_data_from_db(conn, table_name):
    """
    Verilen SQL sorgusunu kullanarak veritabanından verileri çeker 
    ve sonuçları bir pandas DataFrame'ine dönüştürür.

    Parametreler:
    conn: Veritabanı bağlantısını temsil eden psycopg2 bağlantı nesnesi. 
    table_name: Veritabanına gönderilecek tablo ismi.

    Return Değeri:
    DataFrame: Tüm sütunları sayısal ve boş değer içermeyen SQL sorgusunun 
    sonucunu içeren bir pandas DataFrame nesnesi döndürür. Eğer veri setindeki 
    sütunlar sayısal değilse veya boş değer varsa geri dönüş yapılmaz ve hata kaydedilir.
    """
    try:
        logging.info(f"Veritabanından veri çekiliyor: {table_name}")
        
        query = f"SELECT * FROM {table_name};"  
        dataframe = pd.read_sql_query(query, conn)
        
        logging.info(f"Veriler çekildi, kontrol ediliyor...")
        
        print("Veri Türleri:", dataframe.dtypes)  # Sütun türlerini kontrol et
        numeric_columns = dataframe.select_dtypes(include=[np.number])
        empty_check = dataframe.isnull().any().any()
        print(numeric_columns)
        print("Sayısal Sütunlar:", numeric_columns.columns)  # Sayısal sütunları kontrol et
        print("Sayısal Sütun Sayısı:", len(numeric_columns.columns))
        print("Toplam Sütun Sayısı:", len(dataframe.columns))

        return dataframe
        
    except Exception as e:
        logging.error(f"Veri çekme hatası: {e}")
        return None


    
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

if __name__ == "__main__":
    setup_logging()
    logging.info("Kod çalıştırılıyor.")

    data_path = app.data_path  # app modülünden data_path'i al
    table_name = os.path.splitext(os.path.basename(data_path))[0]

    conn = connect_db()

    if conn is not None:
        df = pd.read_csv(data_path)

        create_table(table_name, conn, df)

        insert_data(table_name, df)

        logging.info(f"Tablodan veri çekmeye çalışılıyor: {table_name}")
        df_from_db = get_data_from_db(conn, table_name)

        if df_from_db is not None:
            logging.info(f"{table_name} tablosundan veri başarıyla çekildi.")
        else:
            logging.error(f"{table_name} tablosundan veri alınamadı.")
    else:
        logging.error("Veritabanına bağlanılamadı.")

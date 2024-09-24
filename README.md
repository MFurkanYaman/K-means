### K-Means Kümeleme Projesi (Kütüphane ile ve Kütüphane Olmadan)

Bu projede, K-Means algoritması hem Python kütüphaneleri kullanılarak hem de kütüphane kullanılmadan sıfırdan uygulanmıştır. İki farklı yaklaşım ile verilerin kümelere ayrılması sağlanmıştır. Ayrıca, en iyi küme sayısı ve model performansı **Silhouette Skoru** ile belirlenmiştir.

## Proje Hakkında

**K-Means**, verileri benzerliklerine göre gruplamak için kullanılan bir kümeleme algoritmasıdır. Bu projede iki farklı dosya bulunuyor:
1. **kmeans_with_library.py**: `scikit-learn` kütüphanesi kullanarak K-Means algoritmasını uygular.
2. **kmeans_without_library.py**: Kütüphane kullanmadan, algoritmanın temel adımları sıfırdan uygulanır.

Her iki dosya da SQL veritabanından veri çekme, verileri normalize etme, en uygun küme sayısını bulma ve sonuçları Excel'e yazdırma gibi işlemleri yapar.

## Silhouette Skoru Nedir?

Silhouette skoru, bir kümeleme modelinin ne kadar iyi performans gösterdiğini ölçen bir metriktir. Her bir veri noktası için şu şekilde hesaplanır:

$$
\text{Silhouette Skoru} = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$


- **a(i)**: Aynı küme içindeki diğer noktalara olan ortalama mesafe.
- **b(i)**: En yakın diğer kümeye olan ortalama mesafe.

Silhouette skoru, bu iki değeri kullanarak kümeleme kalitesini değerlendirir. Skor 1'e ne kadar yakınsa, kümeler o kadar başarılı bir şekilde ayrılmıştır.

## Dosya Yapısı

- **kmeans_with_library.py**: `scikit-learn` ile K-Means algoritması ve en iyi küme sayısının bulunması.
- **kmeans_without_library.py**: Sıfırdan K-Means algoritması ve kümeleme yapılması.
- **run_kmeans.py**: İki yaklaşımı da çalıştıran ana dosya.
- **log_for_kmeans_with_library.txt**: Kütüphane ile çalışan algoritmanın log kayıtları.
- **log_for_kmeans_without_library.txt**: Kütüphane olmadan çalışan algoritmanın log kayıtları.
- **kmeans_results.xlsx**: Kümelenmiş veriler ve sonuçların kaydedildiği Excel dosyası.

## Kullanım
1. Projeyi klonlayın:
   ```bash
   git clone https://github.com/MFurkanYaman/K-means.git
   cd K-means
   
2. Gerekli bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
3. run_kmeans.py dosyasını çalıştırın:
   ```bash
   python run_kmeans.py

import numpy as np
import math

#Yapılacaklar : 

#Distance çalışıyor.
#Centorid kümeleme hatalı düzeltildi.
#Yeni centroidler hesaplanıldı.
#Küme kontrol genel döngü eklenecek.
#Best Score bulunacak.
#Kod modüler hale dönüştürülecek ve kütüphaneli koda entegre edilicek.

"""
clusters:  Hangi noktanın hangi kümede bulunduğu bilgisi vardır.
distance_dict:   Öklid sonucu uzaklık değerleri burada.
new_centroids: yapılan kümeleme sonucunda yeni centroids seçimi gerçekleştirildi.


"""



scaled_data=np.array([[1,2,3],[4,5,6],[2,4,8],[5,1,2],[2,2,4],[4,1,3]])
centroids=np.array([[2,4,8],[5,1,2]])

row_num=scaled_data.shape[0]
centroids_num=2


value=[]
distance_dict={}

total=0
find_min=[]
sayac=0
clusters={}

#Bu kısım doğru
# while True:
    
for i in range(centroids_num):          
    for j in range(row_num):
        distance = np.sqrt(np.sum((scaled_data[j] - centroids[i])**2))
        value.append(distance)
    distance_dict[i]=value
    value=[] 

# print(distance_dict)
results = []
index=0

#Hatalı Düzeltildi.


print(distance_dict)
for i in range(row_num):
    min_value=float("inf")
    for j in range(centroids_num):
        
        if distance_dict[j][i]<min_value:

            min_value=distance_dict[j][i]
            min_key=j

        if min_key not in clusters:
            clusters[min_key]=[]

    clusters[min_key].append(i)
    results.append(f"Index {i}: key {min_key} değeri ({min_value}) daha küçük.")



# yeni centroidi yap.

new_centroids=[]
for i in range(centroids_num):

    for inside_value in clusters[i]:
        total+=scaled_data[inside_value]
    # print(len(clusters[i]))
    total//=len(clusters[i])
    new_centroids.append(total)
    total=0

#yeni centroidsler:
centroids = np.array(new_centroids)






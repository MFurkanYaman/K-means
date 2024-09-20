import numpy as np
import math

#Yapılacaklar : 

#Distance çalışıyor.
#Centorid kümeleme hatalı düzeltildi.
#Küme kontrol genel döngü eklenecek.
#Best Score bulunacak.
#Kod modüler hale dönüştürülecek ve kütüphaneli koda entegre edilicek.




scaled_data=np.array([[1,2,3],[4,5,6],[2,4,8],[5,1,2],[2,2,4],[4,1,3]])
centroids=np.array([[2,4,8],[5,1,2]])

row_num=scaled_data.shape[0]
centroids_num=2


value=[]
old_dict={}


find_min=[]
sayac=0
new_dict={}

#Bu kısım doğru
# while True:
    
for i in range(centroids_num):          
    for j in range(row_num):
        distance = np.sqrt(np.sum((scaled_data[j] - centroids[i])**2))
        value.append(distance)
    old_dict[i]=value
    value=[] 

print(old_dict)
results = []
index=0

#Hatalı Düzeltildi.
for i in range(row_num):
    min_value=float("inf")
    for j in range(centroids_num):
        
        if old_dict[j][i]<min_value:

            min_value=old_dict[j][i]
            min_key=j

        if min_key not in new_dict:
            new_dict[min_key]=[]

    new_dict[min_key].append(i)
    results.append(f"Index {i}: key {min_key} değeri ({min_value}) daha küçük.")


for result in results:
    print(result)


print(new_dict)


    



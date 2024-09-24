import subprocess

def run_scripts():
    
    try:
        subprocess.run(['python', 'kmeans_without_library.py'])

        subprocess.run(['python', 'kmeans_with_library.py'])
        
        print("Çıktı Sonuçları Excel Dosyasına Kaydedilmiştir.")

    except Exception as e:
        print(f"Hata ile karşılaşıldı. {e}")


    

if __name__ == "__main__":    
    run_scripts()

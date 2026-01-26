import fitz  # PyMuPDF
import easyocr
import numpy as np
import threading
import queue
import time
import os

# --- AYARLAR ---
pdf_path = "ErgenekonPDF.pdf"
output_txt = "cikti_hizli.txt"
QUEUE_SIZE = 10  # RAM'de sırada bekleyecek maksimum resim sayısı

# Kuyruk oluştur (İşlemci ve GPU arasındaki köprü)
img_queue = queue.Queue(maxsize=QUEUE_SIZE)
stop_event = threading.Event()


def producer_image_loader():
    """
    Bu fonksiyon arka planda çalışır (CPU).
    Sürekli PDF sayfalarını resme çevirip kuyruğa atar.
    GPU'nun beklemesini engeller.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[CPU] Toplam {total_pages} sayfa hazırlanmaya başlandı...")

    for i in range(total_pages):
        try:
            page = doc.load_page(i)
            # Çözünürlük ayarı (2x iyidir, çok yavaşlarsa 1.5 yapın)
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)

            # Numpy dönüşümü
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img_np = np.ascontiguousarray(img_np[..., :3])

            # Hazır resmi kuyruğa at (Kuyruk doluysa yer açılana kadar bekler)
            img_queue.put((i + 1, img_np))

        except Exception as e:
            print(f"[HATA] Sayfa {i + 1} yüklenemedi: {e}")

    # İş bitince sinyal ver
    stop_event.set()
    print("[CPU] Tüm sayfalar kuyruğa yüklendi.")


def consumer_gpu_ocr():
    """
    Bu fonksiyon Ana süreçtir. Sadece GPU ile OCR yapar.
    Resim hazırlamakla vakit kaybetmez, kuyruktan hazır alır.
    """
    # EasyOCR GPU Modu
    print("[GPU] Model yükleniyor...")
    reader = easyocr.Reader(['tr', 'en'], gpu=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("")  # Dosyayı temizle

    processed_count = 0

    while not (stop_event.is_set() and img_queue.empty()):
        try:
            # Kuyruktan resim al (1 saniye bekle, gelmezse döngüyü kontrol et)
            page_num, img_np = img_queue.get(timeout=1)

            # --- OCR BURADA YAPILIYOR ---
            # detail=0: sadece metin, paragraph=True: satırları birleştir
            result = reader.readtext(img_np, detail=0, paragraph=True)

            text_block = f"\n--- Sayfa {page_num} ---\n" + "\n".join(result) + "\n"

            # Yazma işlemi (Diske yazma çok kısadır, bloklamaz)
            with open(output_txt, "a", encoding="utf-8") as f:
                f.write(text_block)

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"[GPU] {processed_count}. sayfa işlendi...")

            # Kuyruğa işin bittiğini bildir
            img_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU HATA] Bir sorun oluştu: {e}")

    print(f"\nİşlem Tamamlandı! Toplam {processed_count} sayfa çevrildi.")


if __name__ == "__main__":
    # 1. CPU İş parçacığını (Thread) başlat
    t = threading.Thread(target=producer_image_loader)
    t.start()

    # 2. Ana GPU işlemini başlat
    # (Thread arka planda çalışırken bu fonksiyon GPU'yu besleyecek)
    consumer_gpu_ocr()

    # Thread'in kapanmasını bekle
    t.join()
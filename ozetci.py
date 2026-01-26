import os
import json
import ollama
import time

# --- AYARLAR ---
ROOT_FOLDER = './clean_texts'  # İçinde txt dosyaları ve alt klasörler olan ana dizin
OUTPUT_FILE = 'sftOutput1.json'
MODEL_NAME = "llama3.1:8b"  # Kullanılacak model (llama3.1:8b vb.)

# Modelinize vereceğimiz sistem talimatı (System Prompt)
# Modeli, gevezelik yapmadan sadece özet çıkarmaya zorluyoruz.
SYSTEM_PROMPT = """
Sen uzman bir özetleme asistanısın.
Sana verilen metni analiz et ve Türkçe olarak, kapsamlı ama net bir özetini çıkar.
Cevabında "İşte özet:" veya "Metnin özeti şudur" gibi giriş cümleleri kullanma.
Doğrudan özeti yaz.
"""


def clean_text(text):
    """Metni basitçe temizler."""
    if not text: return None
    # Unicode hatası önlemi ve boşluk temizliği
    text = text.encode('utf-8', 'ignore').decode('utf-8').strip()
    return " ".join(text.split())


def generate_summary_with_llama(text_content):
    """Ollama üzerinden Llama 3.1'e metni gönderip özet alır."""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': f"Aşağıdaki metni özetle:\n\n{text_content}"
            },
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Model hatası: {e}")
        return None


def main():
    dataset = []
    file_count = 0

    # os.walk ile tüm alt klasörleri gez
    print(f"'{ROOT_FOLDER}' içindeki dosyalar taranıyor...")

    all_files = []
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))

    total_files = len(all_files)
    print(f"Toplam {total_files} adet .txt dosyası bulundu. İşlem başlıyor...\n")

    for index, file_path in enumerate(all_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()

            cleaned_input = clean_text(raw_content)

            # Çok kısa metinleri veya boş dosyaları atla
            if not cleaned_input or len(cleaned_input) < 100:
                print(f"[{index}/{total_files}] Atlandı (Çok kısa): {file_path}")
                continue

            print(f"[{index}/{total_files}] Özetleniyor: {file_path} ...", end="", flush=True)

            # --- LLM İLE ÖZETLEME ---
            summary_output = generate_summary_with_llama(cleaned_input)

            if summary_output:
                # SFT JSON Formatı (Alpaca standardı)
                data_point = {
                    "instruction": "Aşağıdaki metni Türkçe olarak özetle.",
                    "input": cleaned_input,
                    "output": summary_output
                }
                dataset.append(data_point)
                print(" OK.")
            else:
                print(" HATA (Model yanıt vermedi).")

        except Exception as e:
            print(f"\nDosya okuma hatası ({file_path}): {e}")

    # Sonucu Kaydet
    print(f"\nİşlem bitti. Toplam {len(dataset)} veri çifti oluşturuldu.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dosya kaydedildi: {OUTPUT_FILE}")


if __name__ == "__main__":
    if not os.path.exists(ROOT_FOLDER):
        os.makedirs(ROOT_FOLDER)
        print(f"HATA: '{ROOT_FOLDER}' klasörü bulunamadı. Sizin için oluşturdum, lütfen içine txt dosyalarını koyun.")
    else:
        main()
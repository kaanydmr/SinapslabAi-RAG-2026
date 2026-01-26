import os
import json
import ollama

# --- AYARLAR ---
ROOT_FOLDER = './pdf_cleanup'
OUTPUT_FILE = 'dynamic_sft_datasetPDF_fixed.json'
MODEL_NAME = "llama3.2:3b"

# --- DİNAMİK CHUNK AYARLARI ---
CHUNK_SIZE = 8000  # Hedef boyut
MIN_CHUNKS = 5
ABSOLUTE_MAX = 30
OVERLAP = 200  # Örtüşme payı (Bağlam kopmaması için)

SYSTEM_PROMPT = """
Sen uzman bir dava analistisin.
Sana verilen metin parçasını analiz et ve hukuk terminolojisine bağlı kalarak Türkçe olarak net, bilgi yoğun bir özet çıkar.
Giriş cümlesi ("Bu metinde...") kullanma. Doğrudan özetle.
"""


def find_safe_boundary(text, target_index, search_range=500, forward=False):
    """
    Hedef index'in etrafında güvenli bir kesim noktası (boşluk veya nokta) arar.
    Kelime ortasından bölmeyi engeller.
    """
    # Metin sınırlarını kontrol et
    if target_index >= len(text):
        return len(text)
    if target_index <= 0:
        return 0

    # Aranacak bölgeyi belirle
    start = max(0, target_index - search_range)
    end = min(len(text), target_index + search_range)
    snippet = text[start:end]

    # Öncelik: Cümle sonu (. ! ? \n) -> Yoksa Kelime sonu (Boşluk)
    # Geriye doğru arama (default)
    relative_target = target_index - start

    # 1. Önce cümle sonlarına bak (En yakın nokta veya yeni satır)
    for i in range(relative_target, 0, -1):
        if snippet[i] in ['.', '\n', '!', '?']:
            return start + i + 1  # Noktayı da dahil et

    # 2. Cümle sonu bulamazsa en yakın boşluğa bak
    for i in range(relative_target, 0, -1):
        if snippet[i] == ' ':
            return start + i

    # 3. Hiçbir şey bulamazsa mecburen olduğu yerden kes (Çok nadir)
    return target_index


def get_dynamic_chunks(text, chunk_size, max_limit):
    """
    Kelime bütünlüğünü koruyarak dinamik parçalama yapar.
    """
    text_len = len(text)

    # Metin kısaysa direkt dön
    if text_len <= chunk_size:
        return [text]

    total_possible_slots = text_len // chunk_size

    # --- DURUM 1: SIRALI ALIM (Küçük/Orta dosyalar) ---
    if total_possible_slots <= max_limit:
        chunks = []
        start = 0
        while start < text_len:
            # Hedef bitiş noktası
            target_end = start + chunk_size

            # Güvenli bitiş noktasını bul (Kelimeyi bölme)
            safe_end = find_safe_boundary(text, target_end)

            # Eğer ilerleme kaydedemiyorsak döngüyü kır (Sonsuz döngü önlemi)
            if safe_end <= start:
                safe_end = min(start + chunk_size, text_len)

            chunk = text[start:safe_end].strip()
            if len(chunk) > 300:  # Çöp chunkları engelle
                chunks.append(chunk)

            # Bir sonraki parça için başlangıcı ayarla (Overlap payı düşerek)
            start = safe_end - OVERLAP

            # Sona geldiysek bitir
            if safe_end >= text_len:
                break

        return chunks

    # --- DURUM 2: SAMPLING (Çok büyük dosyalar) ---
    else:
        chunks = []

        # 1. BAŞLANGIÇ (Giriş)
        safe_first_end = find_safe_boundary(text, chunk_size)
        chunks.append(text[:safe_first_end])

        # 2. BİTİŞ (Sonuç)
        # Sondan chunk_size kadar geriye git, ama güvenli yerden başla
        target_last_start = text_len - chunk_size
        safe_last_start = find_safe_boundary(text, target_last_start)
        chunks.append(text[safe_last_start:])

        # 3. ORTA PARÇALAR
        remaining_slots = max_limit - 2

        # Orta alanı hesapla
        inner_start = safe_first_end
        inner_end = safe_last_start
        if inner_end > inner_start:
            step = (inner_end - inner_start) // (remaining_slots + 1)

            current_target = inner_start + step

            for _ in range(remaining_slots):
                # Başlangıcı güvenli bir yere çek
                safe_chunk_start = find_safe_boundary(text, current_target)

                # Bitişi güvenli bir yere çek
                safe_chunk_end = find_safe_boundary(text, safe_chunk_start + chunk_size)

                # Sınır kontrolü
                if safe_chunk_end < inner_end and safe_chunk_end > safe_chunk_start:
                    chunk = text[safe_chunk_start:safe_chunk_end].strip()
                    if len(chunk) > 500:
                        chunks.append(chunk)

                current_target += step

        return chunks


def generate_summary(text_chunk):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Aşağıdaki metni özetle:\n\n{text_chunk}"},
            ],
            options={
                'num_ctx': 4096,
                'num_predict': 200,
                'temperature': 0.1,  # Daha tutarlı olması için düşürdüm
                'num_gpu': 99
            }
        )
        return response['message']['content']
    except Exception as e:
        print(f"Ollama Hatası: {e}")
        return None


def main():
    dataset = []
    total_processed_chunks = 0

    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(ROOT_FOLDER) for f in filenames if
                 f.endswith('.txt')]

    print(f"Toplam {len(all_files)} dosya bulundu.")
    print(f"STRATEJİ: Akıllı Bölme + Overlap ({OVERLAP} char)\n")

    for idx, file_path in enumerate(all_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                # Çok fazla bitişik boşluğu tek boşluğa indir ama satır sonlarını koru
                # content = " ".join(content.split()) -> Bu satır yapısını bozabilir, yerine:
                content = " ".join(content.split())

            if len(content) < 500: continue

            chunks = get_dynamic_chunks(content, CHUNK_SIZE, ABSOLUTE_MAX)

            file_name = os.path.basename(file_path)
            file_kb = len(content) / 1024

            print(f"[{idx}] {file_name} ({file_kb:.0f} KB) -> {len(chunks)} parça. ", end="", flush=True)

            file_success_count = 0
            for chunk in chunks:
                summary = generate_summary(chunk)
                if summary:
                    dataset.append({
                        "instruction": "Aşağıdaki hukuki metni analiz et ve özetle.",
                        "input": chunk,
                        "output": summary
                    })
                    file_success_count += 1
                    print(".", end="", flush=True)  # İlerleme çubuğu gibi nokta koy

            print(f" Tamamlandı.")
            total_processed_chunks += file_success_count

        except Exception as e:
            print(f"\nDosya Hatası ({file_path}): {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\nBİTTİ. Toplam {total_processed_chunks} veri üretildi -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
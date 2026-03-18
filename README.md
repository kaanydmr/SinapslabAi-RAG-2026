# RAG + CAG 

**KV Cache + Semantic CAG + RAG Mimarisi** | Qwen3:8B via Ollama | Tamamen Türkçe

Bu sistem, hukuk verilerini üç aşamalı bir pipeline üzerinden işler. Amaç, en doğru cevabı en hızlı şekilde (milisaniyeler içinde) kullanıcıya sunmaktır.

## Mimari Akış

```mermaid
graph TD
    A[Kullanıcı Sorgusu] --> B{1. KV Cache CAG}
    B -- Hit  --> C[Cevap Dön]
    B -- Miss --> D{2. Semantik CAG}
    D -- Hit  --> C
    D -- Miss --> E{3. RAG Pipeline}
    E --> F[Vektör Arama / FAISS]
    F --> G[LLM  / Qwen3]
    G --> H[Doğrulama / Kayıt]
    H --> C
```

## 3 Farklı Mod

1.  **KV Cache CAG (Prompt Pre-fill):** İlk 100 soru-cevap ikilisi Ollama'nın VRAM'inde önceden (pre-filled) olarak tutulur. Eğer soru buradaysa cevap **minimum** gecikmeyle döner.
2.  **Semantik CAG (Semantic Cache):** Geçmişte sorulmuş ve doğrulanmış tüm sorular vektör tabanlı olarak taranır. %90+ benzerlik varsa RAG'e gitmeden cevaplanır.
3.  **RAG Pipeline:** Eğer yukarıdaki iki katmanda cevap bulunamazsa, 50.000+ satırlık veri tabanında FAISS ile arama yapılır ve LLM (Qwen3) bağlamı kullanarak cevap üretir.

## Gereksinimler

- Python 3.9+
- [Ollama](https://ollama.ai/) 
- `qwen3:8b` modeli (`ollama pull qwen3:8b`)

## Kurulum ve Çalıştırma

### 1. Dosya Yapısı
dataset dosyasında bulunan `main set.json` dosyasının proje kök dizininde olduğundan emin olun.

### 2. Bağımlılıklar
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Başlatın
```bash
python app.py
```
Default olarak `http://127.0.0.1:5000` adresinde çalışacaktır.

## Modüller

| Modül | Görev |
|-------|-------|
| `src/kv_cag_pipeline.py` |  VRAM tabanlı KV Cache yönetimi |
| `src/cag_cache.py` | KV cahe and Semantic cache (CAG) |
| `src/rag_pipeline.py` | (RAG)  |
| `src/vector_store.py` | FAISS vektör deposu işlemleri |
| `src/validator.py` | LLM cevaplarının hukuki doğruluğunu kontrol eder |
| `app.py` | Flask API  |

## (UI)

- **KV Cache CAG:** Bu seçenek aktif edildiğinde model belleğe yüklenir (10-30sn sürer) ve statik QA seti anında sorgulanabilir hale gelir.
- **CAG Geliştirme (Refine):** Önbellekten gelen cevapların LLM tarafından yeniden işlenmesini sağlar (doğruluk daha yüksek ama daha yavaş).
- **Semantik Eşik:** Önbellek benzerlik threshold ayarlar.

---


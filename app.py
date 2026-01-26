import os
import torch
import gc
import tempfile
import shutil
import streamlit as st
import ollama
import fitz  # PyMuPDF
import concurrent.futures
from rapidocr_onnxruntime import RapidOCR

# --- LangChain Importları ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

# ==========================================
# AYARLAR (Burayı Kendi Sistemine Göre Düzenle)
# ==========================================
st.set_page_config(page_title="Hukuk Asistanı RAG", layout="wide", page_icon="⚖️")

# Ollama'da 'ollama list' yazdığında çıkan model ismin:
OLLAMA_MODEL_NAME = "qwen2.5-7b-sft"

# GPU Gücüne göre (RTX 5070 Ti için 128 ideal, hata verirse 64 yap)
EMBEDDING_BATCH_SIZE = 128
OCR_WORKERS = 8  # Aynı anda kaç sayfa OCR yapılacak?


# ==========================================
# 1. Donanım ve Bellek Yönetimi
# ==========================================

def clear_gpu_memory():
    """GPU belleğini ve çöp toplayıcıyı temizler."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 Bellek Temizlendi!")


def get_gpu_embedding():
    """PDF işlerken GPU kullanır (Yüksek Batch Size)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Embedding Modeli {device.upper()} modunda başlatılıyor...")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBEDDING_BATCH_SIZE}
    )


def get_cpu_embedding():
    """Sohbet sırasında CPU kullanır (VRAM'i LLM'e bırakır)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4}
    )


# ==========================================
# 2. Paralel PDF ve OCR Motoru
# ==========================================

def process_page_wrapper(args):
    """Tek bir sayfayı işleyen işçi fonksiyon."""
    page_num, file_path, use_gpu = args

    try:
        doc = fitz.open(file_path)
        page = doc[page_num]

        # 1. Metni normal yolla al
        text = page.get_text()
        clean_text = " ".join(text.split())

        # 2. Metin yoksa OCR yap (Resim PDF)
        if len(clean_text) < 20:
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

            # OCR Motorunu Başlat
            if use_gpu:
                ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
            else:
                ocr = RapidOCR()

            ocr_result, _ = ocr(img_bytes)
            if ocr_result:
                clean_text = " ".join([line[1] for line in ocr_result])

        doc.close()

        if clean_text:
            return Document(
                page_content=clean_text,
                metadata={"page": page_num + 1, "source": file_path}
            )
        return None

    except Exception as e:
        print(f"⚠️ Hata (Sayfa {page_num + 1}): {e}")
        return None


def process_pdf_parallel(file_path: str):
    """PDF'i paralel işleyen ana fonksiyon."""
    doc = fitz.open(file_path)
    total_pages = len(doc)
    doc.close()

    use_gpu = torch.cuda.is_available()
    tasks = [(i, file_path, use_gpu) for i in range(total_pages)]
    documents = []

    progress_bar = st.progress(0, text="Belge taranıyor (OCR)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        results = list(executor.map(process_page_wrapper, tasks))

        for i, res in enumerate(results):
            if res:
                documents.append(res)
            progress_bar.progress((i + 1) / total_pages, text=f"Tarama: {i + 1}/{total_pages} sayfa")

    progress_bar.empty()

    # Sayfaları sıraya diz (Threadler karıştırmış olabilir)
    documents.sort(key=lambda x: x.metadata["page"])
    return documents


def ingest_pdf_to_vector_db(file_path):
    """PDF -> Vektör Veritabanı akışı."""
    # 1. Okuma
    raw_docs = process_pdf_parallel(file_path)
    if not raw_docs:
        st.error("PDF'den metin okunamadı!")
        return False

    # 2. GPU Embedding Hazırlığı
    gpu_embeddings = get_gpu_embedding()

    # 3. Parçalama (Chunking)
    text_splitter = SemanticChunker(
        gpu_embeddings,
        breakpoint_threshold_type="percentile"
    )

    with st.spinner("🧠 İçerik vektörleştiriliyor..."):
        chunks = text_splitter.split_documents(raw_docs)

    # 4. Veritabanı Kayıt (Eskisini sil)
    if os.path.exists("./chroma_db_temp"):
        try:
            shutil.rmtree("./chroma_db_temp")
        except:
            pass

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=gpu_embeddings,
        collection_name="optimized_rag",
        persist_directory="./chroma_db_temp"
    )

    st.toast(f"✅ {len(raw_docs)} sayfa, {len(chunks)} parça olarak kaydedildi.", icon="🚀")

    # 5. Temizlik
    del gpu_embeddings
    del vector_db
    del text_splitter
    clear_gpu_memory()

    return True


# ==========================================
# 3. Sorgu ve LLM Motoru
# ==========================================

def get_query_engine():
    """Retrieval motorunu hazırlar."""
    cpu_embeddings = get_cpu_embedding()

    vector_db = Chroma(
        persist_directory="./chroma_db_temp",
        embedding_function=cpu_embeddings,
        collection_name="optimized_rag"
    )

    base_retriever = vector_db.as_retriever(search_kwargs={"k": 15})  # İlk 15 adayı getir
    compressor = FlashrankRerank()  # En iyileri seç

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


def run_llm(context, query):
    # DÜZELTME: "### Instruction" etiketlerini kaldırdık.
    # Çünkü senin Modelfile ayarın bunları zaten otomatik ekliyor.
    # Biz sadece saf veriyi (System ve User) göndereceğiz.

    system_msg = """Sen titiz bir hukuk asistanısın. 
Kurallar:
1. Asla uydurma yapma, sadece aşağıdaki BAĞLAM içindeki bilgileri kullan.
2. Cevabında mutlaka [Sayfa X] referansı ver.
3. Eğer bilgi metinde yoksa "Bu bilgi dosyada yer almıyor" de."""

    user_msg = f"""Aşağıdaki dava dosyası parçasını analiz et ve soruyu cevapla.

BAĞLAM:
{context}

SORU:
{query}
"""

    return ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        stream=True,
        options={
            "temperature": 0.1,  # Mantıklı olması için düşük
            "repeat_penalty": 1.2,  # <--- KRİTİK: Modelin kendini tekrar etmesini yasaklar
            "num_predict": 400,  # <--- KRİTİK: Cevabı en fazla 400 kelimeyle sınırlar (Sonsuz döngüyü keser)
            "stop": ["###", "Instruction:", "Input:", "Response:"]
        }
    )
# ==========================================
# 4. Arayüz (Streamlit)
# ==========================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False

with st.sidebar:
    st.header("📂 Dava Dosyası Yükle")
    uploaded_file = st.file_uploader("PDF Seç", type=["pdf"])

    if uploaded_file and not st.session_state["db_ready"]:
        with st.status("🚀 Analiz Başlatılıyor...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            success = ingest_pdf_to_vector_db(pdf_path)

            if success:
                st.session_state["db_ready"] = True
                status.update(label="✅ Hazır!", state="complete", expanded=False)

    if st.button("🗑️ Sohbeti Sıfırla"):
        st.session_state["messages"] = []
        st.rerun()

st.title("⚖️ Hukuk Asistanı (SFT + RAG)")

# Mesaj Geçmişi
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı Girişi
if prompt := st.chat_input("Dava dosyası hakkında bir soru sor..."):
    if not st.session_state["db_ready"]:
        st.warning("Lütfen önce bir dosya yükleyin.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Retrieval (Arama)
        try:
            retriever = get_query_engine()
            docs = retriever.invoke(prompt)[:7]  # En iyi 7 parça
        except Exception as e:
            st.error(f"Arama motorunda hata oluştu: {e}")
            st.stop()

        # Eğer hiç alakalı döküman bulunamazsa
        if not docs:
            st.warning("⚠️ Bu soruyla ilgili dava dosyasında hiçbir bilgi bulunamadı.")
            full_response = "Bu bilgi mevcut dökümanlarda yer almıyor."
            st.write(full_response)
        else:
            # Context oluştur
            context_text = "\n\n".join([f"[Sayfa {d.metadata.get('page', '?')}]: {d.page_content}" for d in docs])

            # 2. Generation (Üretim)
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = run_llm(context_text, prompt)
                for chunk in stream:
                    content = chunk['message']['content']
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Model cevap üretirken hata oluştu: {e}")

            # Kaynakları göster
            with st.expander("🔍 İncelenen Deliller/İfadeler"):
                st.write(context_text)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
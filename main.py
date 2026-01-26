import os
import torch
import gc
import tempfile
import shutil
import streamlit as st
import ollama
from langchain_classic.retrievers import ContextualCompressionRetriever

# --- LangChain & Processing Imports ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
# --- Reranking & Retrieval Importları (DÜZELTİLMİŞ HALİ) ---
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

import os
import torch
import gc
import tempfile
import shutil
import streamlit as st
import ollama

# --- LangChain Importları ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

# -----------------------------
# 1. Donanım ve Temizlik Ayarları
# -----------------------------
st.set_page_config(page_title="Smart VRAM RAG", layout="wide", page_icon="🧠")


def clear_gpu_memory():
    """GPU belleğini zorla boşaltır."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("🧹 GPU Belleği Temizlendi!")


# -----------------------------
# 2. Embedding Modelleri (GPU ve CPU Ayrımı)
# -----------------------------

def get_gpu_embedding():
    """
    PDF işlerken kullanılır. GPU'yu sömürür, işi hızlı bitirir.
    DİKKAT: Bunu cache'lemiyoruz, işi bitince sileceğiz.
    """
    print("⚡ Embedding Modeli GPU'ya Yükleniyor...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},  # Zorla GPU
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )


@st.cache_resource(show_spinner=False)
def get_cpu_embedding():
    """
    Sohbet sırasında kullanılır. VRAM harcamaz, RAM kullanır.
    """
    print("🐢 Embedding Modeli CPU'ya Yükleniyor (Sohbet Modu)...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},  # Zorla CPU
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4}
    )


# -----------------------------
# 3. İşlem Fonksiyonları
# -----------------------------

def process_pdf_fast(file_path: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())
    return docs


def ingest_pdf_to_vector_db(file_path):
    """
    Bu fonksiyon:
    1. GPU Embedding yükler.
    2. Vektörleri oluşturur.
    3. GPU Embedding modelini YOK EDER.
    """
    # 1. PDF Oku
    raw_docs = process_pdf_fast(file_path)

    # 2. GPU Modelini Yükle
    gpu_embeddings = get_gpu_embedding()

    # 3. Semantic Chunking (GPU ile hızlıca)
    text_splitter = SemanticChunker(
        gpu_embeddings,
        breakpoint_threshold_type="percentile"
    )

    with st.spinner("GPU ile parçalanıyor ve vektörleştiriliyor..."):
        chunks = text_splitter.split_documents(raw_docs)

    # 4. Veritabanına Yaz (Diske Kaydet)
    # Eğer öncekileri silmek istersen bu bloğu aç:
    if os.path.exists("./chroma_db_temp"):
        shutil.rmtree("./chroma_db_temp")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=gpu_embeddings,
        collection_name="optimized_rag",
        persist_directory="./chroma_db_temp"
    )

    st.toast(f"✅ {len(raw_docs)} sayfa işlendi. Şimdi VRAM temizleniyor...", icon="🧹")

    # 5. KRİTİK ADIM: GPU Modelini ve Veritabanı bağlantısını öldür
    del gpu_embeddings
    del vector_db
    del text_splitter
    clear_gpu_memory()  # VRAM'i Ollama'ya geri veriyoruz

    return True


def get_query_engine():
    """
    Sorgu motorunu CPU embedding ile hazırlar.
    """
    # CPU embedding modelini al (VRAM harcamaz)
    cpu_embeddings = get_cpu_embedding()

    # Diskteki veritabanını CPU modeliyle tekrar aç
    vector_db = Chroma(
        persist_directory="./chroma_db_temp",
        embedding_function=cpu_embeddings,
        collection_name="optimized_rag"
    )

    # Retrieval Ayarları
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 15})

    # Reranker (Otomatik model indirmeli)
    compressor = FlashrankRerank()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever


def run_llm(context, query):
    system_prompt = (
        "Sen yardımsever bir asistansın. Verilen bağlama göre cevap ver.\n"
        "Kurallar: [Sayfa X] referansı ver. Bilmiyorsan bilmiyorum de."
    )
    user_prompt = f"BAĞLAM:\n{context}\n\nSORU: {query}"

    return ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True
    )


# -----------------------------
# 4. Arayüz (Streamlit)
# -----------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "db_ready" not in st.session_state:
    st.session_state["db_ready"] = False

with st.sidebar:
    st.header("📂 Dosya Yükleme")
    uploaded_file = st.file_uploader("PDF Yükle", type=["pdf"])

    if uploaded_file and not st.session_state["db_ready"]:
        with st.status("🚀 İşleniyor (GPU Modu)...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Ağır işi yap ve VRAM'i temizle
            ingest_pdf_to_vector_db(pdf_path)

            st.session_state["db_ready"] = True
            status.update(label="✅ Hazır! GPU Boşaltıldı.", state="complete", expanded=False)

    if st.button("Sohbeti Sıfırla"):
        st.session_state["messages"] = []
        st.rerun()

st.title("🧠 VRAM-Saver RAG")

# Mesajları Göster
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı Sorgusu
if prompt := st.chat_input("Sorunu sor..."):
    if not st.session_state["db_ready"]:
        st.error("Önce PDF yükle.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieval (CPU Embedder kullanır)
        retriever = get_query_engine()
        docs = retriever.invoke(prompt)[:5]

        context_text = "\n\n".join([f"[Sayfa {d.metadata.get('page', '?')}]: {d.page_content}" for d in docs])

        # Generation (LLM - GPU kullanır, çünkü Embedder sildiğimiz için yer var)
        response_placeholder = st.empty()
        full_response = ""
        stream = run_llm(context_text, prompt)

        for chunk in stream:
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        with st.expander("Kaynaklar"):
            st.write(context_text)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
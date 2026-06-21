import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# --- UI Configuration & Styling ---
st.set_page_config(page_title="RAG Document Assistant", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stChatFloatingInputContainer { background-color: rgba(0,0,0,0); }
    .metric-box {
        background-color: #1e293b;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        color: #38bdf8;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Enterprise RAG Analyst")
st.caption("Local production-ready RAG pipeline for secure document analysis.")

PDF_PATH = "test.pdf"

# --- Core RAG Pipeline Initialization ---
@st.cache_resource
def initialize_rag():
    if not os.path.exists(PDF_PATH):
        return None, None
        
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = text_splitter.split_documents(docs)

    model_name = "BAAI/bge-small-en-v1.5"
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs=encode_kwargs
    )
    
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = OllamaLLM(model="llama3")
    return retriever, llm

retriever, llm = initialize_rag()

# --- Chat Session Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not os.path.exists(PDF_PATH):
    st.error(f"Error: '{PDF_PATH}' not found. Please place the file in the root directory.")
else:
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ System Metrics")
        st.success("✅ 'test.pdf' Indexed")
        st.info("Architecture: LCEL Pipeline\n\nVector Store: ChromaDB\n\nLLM: Llama 3")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Render Message History ---
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    # --- Chat Agent Input & Execution ---
    if user_query := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.write(user_query)
        
        start_time = time.time()

        # Context retrieval
        relevant_docs = retriever.invoke(user_query)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
        
        def format_chat_history(history):
            formatted = ""
            for msg in history:
                prefix = "Kullanıcı" if isinstance(msg, HumanMessage) else "Asistan"
                formatted += f"{prefix}: {msg.content}\n"
            return formatted

        # --- TURKISH RESPONSE INSTRUCTION ---
        system_prompt_template = (
            "Sen profesyonel bir teknik döküman asistanısın. Aşağıda sana verilen sohbet geçmişini ve dökümandan alınan bağlamı "
            "kullanarak kullanıcının sorusuna net, detaylı ve tamamen TÜRKÇE bir cevap ver. Eğer sorulan sorunun cevabı döküman "
            "içeriğinde (context) kesin olarak bulunmuyorsa, dökümanda bu bilgiye ulaşamadığını belirt. Kafandan bilgi uydurma.\n\n"
            "Sohbet Geçmişi:\n{chat_history}\n"
            "Dökümandan Alınan Bağlam (Context):\n{context}\n\n"
            "Soru: {input}"
        )
        
        chat_history_str = format_chat_history(st.session_state.chat_history)
        final_prompt = system_prompt_template.format(
            chat_history=chat_history_str,
            context=context_text,
            input=user_query
        )

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in llm.stream(final_prompt):
                    full_response += chunk
                    response_placeholder.write(full_response + "▌")
                
                response_placeholder.write(full_response)
                
                # Performance profiling
                end_time = time.time()
                elapsed_time_ms = int((end_time - start_time) * 1000)
                elapsed_time_sec = round(end_time - start_time, 2)
                
                st.markdown(f"""
                    <div class="metric-box">
                        ⏱️ Latency: {elapsed_time_ms} ms (~{elapsed_time_sec} seconds)
                    </div>
                """, unsafe_allow_html=True)
                
                # Document lineage tracing
                with st.expander("🔍 Retrieved Document Segments (Context Tracing)"):
                    for i, doc in enumerate(relevant_docs):
                        page_num = doc.metadata.get("page", 0) + 1
                        st.markdown(f"📍 **Source {i+1} — Page: {page_num}**")
                        st.info(doc.page_content)
                
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=full_response))
                
            except Exception as e:
                st.error(f"Execution Error: {e}")
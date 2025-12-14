import streamlit as st

import os
import torch
import warnings

# --------------------------------------------------------- 
# JAUNIE IMPORTI 
# --------------------------------------------------------- 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma  # Fixed import: Use community version instead of langchain_chroma 
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig 
from transformers.utils import logging  # Added to suppress warnings 
from langchain_huggingface import HuggingFacePipeline 
from sentence_transformers import CrossEncoder 
from huggingface_hub import login 

warnings.filterwarnings("ignore") 

logging.set_verbosity_error() 

# --------------------------------------------------------- 
# KONFIGURĀCIJA 
# --------------------------------------------------------- 

# Pārbaudam GPU 
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"✅ Using device: {device}") 

# Ceļi uz datiem 
DATA_PATH = "data/pdfs" 
DB_PATH = "data/vector_stores" 

# Specific configuration based on user preference 
CHUNK_CONFIG = {"chunk_size": 800, "chunk_overlap": 100, "name": "Small"} 
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
TOP_N_CANDIDATES = 50 
TARGET_K_FOR_RAR = 5 

# Embedding un Reranker modeļi 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

# --------------------------------------------------------- 
# FUNKCIJAS 
# --------------------------------------------------------- 

@st.cache_resource
def load_documents():
    """Ielādē visus PDF no mapes.""" 
    docs = [] 
    if not os.path.exists(DATA_PATH): 
        os.makedirs(DATA_PATH) 
        st.warning(f"Lūdzu ievietojiet PDF failus mapē: {DATA_PATH}") 
        return [] 
    
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')] 
    if not files: 
        st.warning("Mape data/pdfs ir tukša!") 
        return [] 

    for filename in files: 
        try: 
            loader = PyPDFLoader(os.path.join(DATA_PATH, filename)) 
            docs.extend(loader.load()) 
        except Exception as e: 
            st.error(f"Error loading {filename}: {e}") 
    return docs 

@st.cache_resource
def create_vector_store(documents, config):
    """Izveido ChromaDB indeksu ar specificēto chunk stratēģiju.""" 
    # Inicializējam embeddings uz GPU 
    model_kwargs = {'device': device} 
    encode_kwargs = {'normalize_embeddings': False} 
    embeddings = HuggingFaceEmbeddings( 
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 
    ) 
    
    
    splitter = RecursiveCharacterTextSplitter( 
        chunk_size=config['chunk_size'], 
        chunk_overlap=config['chunk_overlap'] 
    ) 
    split_docs = splitter.split_documents(documents) 
    
    persist_dir = os.path.join(DB_PATH, config['name']) 
    
    # Izveidojam vai ielādējam Chroma db 
    vectorstore = Chroma.from_documents( 
        documents=split_docs, 
        embedding=embeddings, 
        persist_directory=persist_dir, 
        collection_name=f"collection_{config['name']}" 
    ) 
    return vectorstore 

@st.cache_resource
def get_llm_pipeline(model_id):
    """Ielādē LLM ar kvantizāciju un atgriež pipeline un tokenizer.""" 
    # Login to Hugging Face 
    if os.path.exists("hf_token.txt"): 
        with open("hf_token.txt", "r") as f: 
            token = f.read().strip() 
        login(token) 
    else: 
        st.warning("hf_token.txt not found, proceeding without login") 

    st.info(f"Loading Model: {model_id}...") 
    
    bnb_config = BitsAndBytesConfig( 
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_use_double_quant=True, 
    ) 

    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
        
    model = AutoModelForCausalLM.from_pretrained( 
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto" 
    ) 

    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512, 
        temperature=0.1, 
        repetition_penalty=1.1, 
        return_full_text=False, 
        batch_size=24 
    ) 
    
    return HuggingFacePipeline(pipeline=pipe), tokenizer 

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME, device=device) 

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs) 

def retrieve_and_rerank(query, vectorstore, top_n, top_k): 
    """RaR loģika: Paņem N kandidātus, pārkārto ar CrossEncoder, atgriež Top K.""" 
     
    candidates = vectorstore.similarity_search(query, k=top_n) 
    
    if not candidates: 
        return [] 

    doc_texts = [d.page_content for d in candidates] 
    sentence_pairs = [[query, doc_text] for doc_text in doc_texts] 
    
    reranker_model = load_reranker() 
    scores = reranker_model.predict(sentence_pairs) 
    
    results_with_scores = sorted( 
        zip(candidates, scores),  
        key=lambda x: x[1],  
        reverse=True 
    ) 
    
    top_docs = [doc for doc, score in results_with_scores[:top_k]] 
    return top_docs 

# --------------------------------------------------------- 
# STREAMLIT APP 
# --------------------------------------------------------- 

st.set_page_config( 
    page_title="Pilienu valodas modeļu tērzētāva", 
    layout="centered" 
) 
st.title("Pilienu valodas modeļu tērzētāva") 

raw_docs = load_documents() 
if raw_docs: 
    vector_store = create_vector_store(raw_docs, CHUNK_CONFIG) 
    llm, tokenizer = get_llm_pipeline(MODEL_ID) 

    template = """You are an expert computer science assistant. Answer the question based ONLY on the following context. 

Context: 
{context} 

Question: {question} 

Detailed Answer:""" 
    rag_prompt = PromptTemplate.from_template(template) 
else: 
    st.error("No documents loaded. Cannot proceed with RAG.") 

if "messages" not in st.session_state: 
    st.session_state.messages = [] 

def get_llm_response(user_input): 
    if not raw_docs: 
        return "No documents available for RAG." 
    
    # Use RaG + RaR 
    top_docs = retrieve_and_rerank(user_input, vector_store, TOP_N_CANDIDATES, TARGET_K_FOR_RAR) 
    context_str = format_docs(top_docs) 
    formatted_prompt = rag_prompt.format(context=context_str, question=user_input) 
    
    # Generate response 
    response = llm.invoke(formatted_prompt) 
    return response.strip() 

# Display chat messages from history on app rerun 
for message in st.session_state.messages: 
    with st.chat_message(message["role"]): 
        st.markdown(message["content"]) 
        

# React to user input 
if prompt := st.chat_input("Ask me anything:"): 
    st.chat_message("user").markdown(prompt) 
    st.session_state.messages.append({"role": "user", "content": prompt}) 
    llm_response = get_llm_response(prompt) 
    with st.chat_message("assistant"): 
        st.markdown(llm_response) 
    st.session_state.messages.append({"role": "assistant", "content": llm_response}) 
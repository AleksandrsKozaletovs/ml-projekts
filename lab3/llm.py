    import os
import torch
import warnings
from typing import List, Dict

# ---------------------------------------------------------
# IMPORTI 
# ---------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from transformers.utils import logging  
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import CrossEncoder
from huggingface_hub import login

# Ignorējam brīdinājumus, lai tīrāka izvade
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
RESULTS_FILE = "results.txt"

# Parametri: Fragmentu (Chunk) stratēģijas
CHUNK_CONFIGS = [
    {"chunk_size": 800, "chunk_overlap": 100, "name": "Small"},
    {"chunk_size": 1200, "chunk_overlap": 200, "name": "Medium"},
    {"chunk_size": 1600, "chunk_overlap": 300, "name": "Large"}
]

# TEORĒTISKIE JAUTĀJUMI 
QUESTIONS = [
    # Jautājums 1: Transformer Modelis
    "Why is an IDS solution less effective in critical infrastructure than an IPS solution?",
    
    # Jautājums 2: Kvantizācija
    "Based on CERT.LV data, what is the major cybersecurity risk? Name the reason, based on numerical data",
    
    # Jautājums 3: Edge Ierīces
    "Provide a detailed explanation of how a “black hole” attack is carried out, describe its key characteristics, and analyze the risks and potential impacts it poses to companies",
    
    # Jautājums 4: Privātums un Drošība
    "Describe the types of IDS/IPS, how they differ, and for what purposes they are needed?",
    
    # Jautājums 5: GPU apstrāde
    "What methods do IDS and IPS solutions use to detect and identify attack risks?"
    
]

# Modeļu saraksts
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-3B-Instruct"
    #"mistralai/Mistral-7B-v0.3"
    #"Qwen/Qwen2.5-14B-Instruct"
]

# RAG Parametri
TOP_K_VALUES = [3, 5, 7]
TOP_N_CANDIDATES_VALUES = [10, 20, 50]
TARGET_K_FOR_RAR_VALUES = [3, 5, 7]  # Jauns: Variē fināla k pēc rerank

# Embedding un Reranker modeļi
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------
# FUNKCIJAS
# ---------------------------------------------------------

def load_documents():
    """Ielādē visus PDF no mapes."""
    docs = []
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️ Lūdzu ievietojiet PDF failus mapē: {DATA_PATH}")
        return []
    
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not files:
        print("⚠️ Mape data/pdfs ir tukša!")
        return []

    print(f"📂 Loading {len(files)} PDFs...")
    for filename in files:
        try:
            loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return docs

def create_vector_stores(documents):
    """Izveido 3 dažādus ChromaDB indeksus atkarībā no chunk stratēģijas."""
    # Inicializējam embeddings uz GPU
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    stores = {}
    
    for config in CHUNK_CONFIGS:
        print(f"⚙️  Creating Index: {config['name']} ({config['chunk_size']}/{config['chunk_overlap']})")
        
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
        stores[config['name']] = vectorstore
        
    return stores

def get_llm_pipeline(model_id):
    """Ielādē LLM ar kvantizāciju un atgriež pipeline un tokenizer."""
    # Login to Hugging Face
    if os.path.exists("hf_token.txt"):
        with open("hf_token.txt", "r") as f:
            token = f.read().strip()
        login(token)
        print("✅ Logged in to Hugging Face")
    else:
        print("⚠️ hf_token.txt not found, proceeding without login")

    print(f"🚀 Loading Model: {model_id}...")
    
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
        max_new_tokens=512, # samazināts garums atbildei
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=False,
        batch_size=24
    )
    
    return HuggingFacePipeline(pipeline=pipe), tokenizer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Reranker ielāde (globāli, lai neielādētu katru reizi)
print("⚖️  Loading Reranker Model...")
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device)

def retrieve_and_rerank(query, vectorstore, top_n, top_k):
    """RaR loģika: Paņem N kandidātus, pārkārto ar CrossEncoder, atgriež Top K."""
    candidates = vectorstore.similarity_search(query, k=top_n)
    
    if not candidates:
        return []

    doc_texts = [d.page_content for d in candidates]
    sentence_pairs = [[query, doc_text] for doc_text in doc_texts]

    scores = reranker_model.predict(sentence_pairs)

    results_with_scores = sorted(
        zip(candidates, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    top_docs = [doc for doc, score in results_with_scores[:top_k]]
    return top_docs

# ---------------------------------------------------------
# GALVENĀ IZPILDE
# ---------------------------------------------------------

def main():
    # 1. Datu ielāde
    raw_docs = load_documents()
    if not raw_docs: return

    # 2. Indeksu izveide
    vector_stores = create_vector_stores(raw_docs)
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("=== RAG & RaR EXPERIMENT RESULTS ===\n\n")

        for model_name in MODELS_TO_TEST:
            f.write(f"\n\n################ MODEL: {model_name} ################\n")
            
            try:
                llm, tokenizer = get_llm_pipeline(model_name)
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                continue

            template = """You are an expert computer science assistant. Answer the question based ONLY on the following context.

Context:
{context}

Question: {question}

Detailed Answer:"""
            rag_prompt = PromptTemplate.from_template(template)

            # Savāksim visus promptus batch apstrādei
            all_prompts = []
            all_types = []  # Lai izsekotu, kāda tipa atbilde tas ir (vanilla, RAG, RaR)
            all_labels = []  # Etiķetes rakstīšanai failā

            for q_idx, question in enumerate(QUESTIONS):
                print(f"📝 Collecting prompts for Question {q_idx+1}...")

                # --- VANILLA LLM ---
                try:
                    # Instrukciju formāts, izmantojot chat template
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    all_prompts.append(prompt)
                    all_types.append("vanilla")
                    all_labels.append((q_idx, "VANILLA LLM (No Context)"))
                except Exception as e:
                    f.write(f"\n\n{'='*20} QUESTION {q_idx+1} {'='*20}\nText: {question}\n\n[TYPE: VANILLA LLM (No Context)]\nError: {e}\n")

                # Iterējam cauri chunk stratēģijām
                for chunk_config in CHUNK_CONFIGS:
                    store_name = chunk_config['name']
                    vectorstore = vector_stores[store_name]

                    # --- STANDARD RAG ---
                    for k in TOP_K_VALUES:
                        try:
                            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                            context_docs = retriever.invoke(question)
                            context_str = format_docs(context_docs)
                            formatted_prompt = rag_prompt.format(context=context_str, question=question)
                            all_prompts.append(formatted_prompt)
                            all_types.append("rag")
                            all_labels.append((q_idx, f"CHUNK STRATEGY: {store_name} ({chunk_config['chunk_size']}) | [TYPE: RAG | Top_K={k}]"))
                        except Exception as e:
                            f.write(f"\n\n{'='*20} QUESTION {q_idx+1} {'='*20}\nText: {question}\n\n   >>> CHUNK STRATEGY: {store_name} ({chunk_config['chunk_size']})\n\n   [TYPE: RAG | Top_K={k}]\nError: {e}\n")

                    # --- RAG + RaR ---
                    for target_k_for_rar in TARGET_K_FOR_RAR_VALUES:  # Jauns loop pār fināla k
                        for n_cand in TOP_N_CANDIDATES_VALUES:
                            if n_cand <= target_k_for_rar: continue  # Tikai ja kandidāti > fināla k

                            try:
                                top_docs = retrieve_and_rerank(question, vectorstore, n_cand, target_k_for_rar)
                                context_str = format_docs(top_docs)
                                formatted_prompt = rag_prompt.format(context=context_str, question=question)
                                all_prompts.append(formatted_prompt)
                                all_types.append("rar")
                                all_labels.append((q_idx, f"CHUNK STRATEGY: {store_name} ({chunk_config['chunk_size']}) | [TYPE: RAG+RaR | Candidates={n_cand} -> Top_K={target_k_for_rar}]"))
                            except Exception as e:
                                f.write(f"\n\n{'='*20} QUESTION {q_idx+1} {'='*20}\nText: {question}\n\n   >>> CHUNK STRATEGY: {store_name} ({chunk_config['chunk_size']})\n\n   [TYPE: RAG+RaR | Candidates={n_cand} -> Top_K={target_k_for_rar}]\nError: {e}\n")

            # Tagad izpildām batch visiem promptiem
            if all_prompts:
                print(f"🚀 Processing {len(all_prompts)} prompts in batch...")
                try:
                    responses = llm.batch(all_prompts)
                except Exception as e:
                    print(f"❌ Batch error: {e}")
                    responses = ["Error in batch" for _ in all_prompts]

                # Rakstām rezultātus failā, grupējot pēc jautājumiem
                current_q_idx = -1
                for (q_idx, label), response in zip(all_labels, responses):
                    if q_idx != current_q_idx:
                        f.write(f"\n\n{'='*20} QUESTION {q_idx+1} {'='*20}\nText: {QUESTIONS[q_idx]}\n")
                        current_q_idx = q_idx
                    f.write(f"\n   >>> {label}\n")
                    f.write(response.strip() + "\n")

    print(f"\n✅ Done! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
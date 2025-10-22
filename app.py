import streamlit as st
import os
from dotenv import load_dotenv

# CORRECTED AND UPDATED IMPORTS (Fixes all ModuleNotFoundError issues)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS                  
from langchain.chains import RetrievalQA                            
from langchain_core.prompts import PromptTemplate                   
from langchain_community.embeddings import HuggingFaceEmbeddings # Local Embedding Model for Quota Bypass

# 1. API Key Loading
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Bu uyarı Streamlit Cloud'da görünür
    st.error("GEMINI_API_KEY not found. Please check your Secrets.")
    st.stop()

# 2. RAG Pipeline Setup
@st.cache_resource
def setup_rag_pipeline():
    """Sets up the RAG chain, using a local model for embedding to avoid quota issues."""
    
    # 2.1 Data Loading (Reads the 'case_summaries.txt' file)
    try:
        with open("case_summaries.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.warning("case_summaries.txt not found. Using placeholder data for setup.")
        raw_text = "Article 8: Placeholder content. Article 10: Placeholder content."

    # 2.2 Text Splitter (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    # 2.3 Embedding (QUOTA BYPASS: Local Hugging Face Model)
    # Bu kısım, Google'ın kota hatasını (429) atlatır.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Local Embedding Model Loaded.")

    # 2.4 Vector Database (FAISS) Creation
    db = FAISS.from_texts(texts, embeddings)
    
    # 2.5 PROMPT ENGINEERING
    template = """
    You are a Legal Argument Assistant specializing in ECHR precedents. Analyze the 'ARGUMENT' using ONLY the 'CONTEXT'. 
    Your response MUST: 1. Act as a legal professional. 2. Summarize the MOST RELEVANT precedent. 3. Mention the relevant ECHR Article (e.g., Article 8). 4. Keep it under 200 words.

    CONTEXT: {context}
    ARGUMENT: {question}
    Legal Analysis and Precedent Summary:
    """
    RAG_PROMPT_TEMPLATE = PromptTemplate(template=template, input_variables=["context", "question"])

    # 2.6 RAG Chain (RetrievalQA) Setup
    # Cevap üretimi için en hızlı model kullanılıyor
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": RAG_PROMPT_TEMPLATE}
    )
    
    return qa_chain

# RAG Chain Initialization
try:
    qa_chain = setup_rag_pipeline()
except Exception as e:
    # Bu hata genellikle kütüphane eksikliğini yakalar.
    st.error(f"RAG Setup Error: {e}")
    qa_chain = None

# 3. Streamlit Interface (Frontend)
st.set_page_config(page_title="Human-Rights-Casebot", layout="wide") 

st.title("⚖️ Human-Rights-Casebot") 
st.markdown("---")
st.subheader("Paste Your Legal Argument Here")
st.info("This assistant analyzes your argument and retrieves the most relevant ECHR precedent from its database.")

# User input
user_argument = st.text_area(
    "Please enter the legal argument or question you want to be analyzed:",
    height=200,
    placeholder="Example: 'Does the continued retention of biometric data...?'"
)

# Analysis button
if st.button("Analyze and Find Precedent", type="primary"):
    if user_argument:
        with st.spinner("Analyzing Your Legal Argument..."):
            try:
                # GERÇEK API ÇAĞRISI BURAYA GİDİYOR
                if qa_chain is None:
                    st.error("RAG system is not initialized. Check logs for setup errors.")
                else:
                    result = qa_chain.invoke({"query": user_argument})
                    
                    st.subheader("✅ ECHR Precedent Analysis")
                    st.markdown(result['result'])
                
            except Exception as e:
                # API Zaman Aşımı (Timeout) hatasını yakalar.
                st.error(f"An error occurred during response generation: {e}")
                st.warning("Error suggests API Time-Out. Please re-run the app with a new API key.")
    else:
        st.warning("Please enter a legal argument to analyze.")

st.markdown("---")
st.caption("Project Name: Human-Rights-Casebot | Developer: [Your GitHub ID] | GAIH GenAI Bootcamp")



            
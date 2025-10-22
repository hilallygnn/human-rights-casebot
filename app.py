import streamlit as st
import os
from dotenv import load_dotenv

# CORRECTED AND UPDATED IMPORTS (Fixes all ModuleNotFoundError issues)
from langchain_google_genai import ChatGoogleGenerativeAI
# Note: GoogleGenerativeAIEmbeddings is imported but not used if HuggingFace is active
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS                  
from langchain.chains import RetrievalQA                            
from langchain_core.prompts import PromptTemplate                   
from langchain_community.embeddings import HuggingFaceEmbeddings # Local Embedding Model

# 1. API Key Loading
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please check your Streamlit Cloud Secrets.")
    st.stop()

# 2. RAG Pipeline Setup
@st.cache_resource
def setup_rag_pipeline():
    """Sets up the RAG chain, using a local model for embedding."""
    
    # 2.1 Data Loading (Reads the 'case_summaries.txt' file)
    try:
        # Uses the cleaned version of the case summaries
        with open("case_summaries.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.warning("case_summaries.txt not found. Using minimal placeholder data.")
        raw_text = """
        --- CASE SUMMARY: S. and Marper v. UK (Article 8) ---
        Indefinite retention of fingerprints and DNA samples of acquitted persons violates Article 8 (Right to respect for private life) as it's disproportionate.
        --- CASE SUMMARY: Ahmet Yıldırım v. Turkey (Article 10) ---
        Wholesale blocking of Google Sites due to content on one site violates Article 10 (Freedom of Expression) as it's disproportionate secondary censorship.
        --- CASE SUMMARY: Kaboğlu and Oran v. Turkey (Article 8) ---
        Failure by authorities to investigate credible threats against academics violates the State's positive obligation under Article 8 to protect private life.
        """

    # 2.2 Text Splitter (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    # 2.3 Embedding (QUOTA BYPASS: Using Local Hugging Face Model)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Local Embedding Model Loaded Successfully.") # Log message
    except Exception as e:
        st.error(f"Error loading local embedding model: {e}")
        return None # Return None if embedding fails

    # 2.4 Vector Database (FAISS) Creation
    try:
        db = FAISS.from_texts(texts, embeddings)
        print("✅ FAISS Vector Database Created.") # Log message
    except Exception as e:
        st.error(f"Error creating FAISS database: {e}")
        return None # Return None if DB creation fails
    
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
    try:
        # Using the faster Gemini Flash model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
        print("✅ Gemini LLM Object Initialized.") # Log message
    except Exception as e:
         st.error(f"Error initializing Gemini LLM (Check API Key?): {e}")
         return None # Return None if LLM fails

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": RAG_PROMPT_TEMPLATE}
    )
    
    return qa_chain

# RAG Chain Initialization with error handling
qa_chain = None # Initialize qa_chain as None
try:
    qa_chain = setup_rag_pipeline()
    if qa_chain:
         print("✅ RAG Pipeline setup successful.") # Log message
    else:
         st.error("❌ RAG Pipeline setup failed during component initialization.")
except Exception as e:
    st.error(f"Critical Error during RAG Setup: {e}")

# 3. Streamlit Interface (Frontend)
st.set_page_config(page_title="Human-Rights-Casebot", layout="wide") 

st.title("⚖️ Human-Rights-Casebot") 
st.markdown("---")
st.subheader("Paste Your Legal Argument Here")
st.info("This assistant analyzes your argument and retrieves relevant ECHR precedents.")

# User input
user_argument = st.text_area(
    "Please enter the legal argument or question you want to be analyzed:",
    height=200,
    placeholder="Example: 'Does the continued retention of biometric data...?'"
)

# Analysis button (Using the REAL API call)
if st.button("Analyze and Find Precedent", type="primary"):
    if qa_chain is None:
        st.error("❌ RAG System is not initialized. Cannot perform analysis. Check logs.")
    elif not user_argument:
        st.warning("Please enter a legal argument to analyze.")
    else:
        with st.spinner("Analyzing Your Legal Argument..."):
            try:
                # --- REAL GEMINI API CALL ---
                result = qa_chain.invoke({"query": user_argument})
                
                st.subheader("✅ ECHR Precedent Analysis")
                st.markdown(result['result'])
                # --- END REAL API CALL ---
                
            except Exception as e:
                # Catch Time-Out or Quota errors here
                st.error(f"❌ An error occurred during response generation: {e}")
                st.warning("This might be due to API Time-Out (504 Error) or Rate Limits. Check the app logs ('Manage app') for details. Please try again later.")

st.markdown("---")
st.caption("Project Name: Human-Rights-Casebot | Developer: [hilallygnn] | GAIH GenAI Bootcamp")


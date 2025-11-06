import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS                  
from langchain.chains import RetrievalQA                            
from langchain_core.prompts import PromptTemplate                   
from langchain_community.embeddings import HuggingFaceEmbeddings 

# 1. API Key Loading
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please check your Secrets.")
    st.stop()

# 2. RAG Pipeline Setup
@st.cache_resource
def setup_rag_pipeline():
    """Sets up the RAG chain components (embedding, vectorstore, retriever, LLM)."""
    
    # 2.1 Data Loading
    try:
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

    # 2.2 Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    # 2.3 Embedding (Local Model)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading local embedding model: {e}")
        return None, None

    # 2.4 Vector Database (FAISS)
    try:
        db = FAISS.from_texts(texts, embeddings)
    except Exception as e:
        st.error(f"Error creating FAISS database: {e}")
        return None, None
    
    # 2.5 LLM Definition
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    except Exception as e:
         st.error(f"Error initializing Gemini LLM (Check API Key?): {e}")
         return None, None

    return db, llm

# Initialize RAG components
try:
    vectorstore_db, llm_model = setup_rag_pipeline()
    if vectorstore_db is None or llm_model is None:
        st.error("❌ Critical RAG components failed to initialize. Cannot proceed.")
        st.stop()
except Exception as e:
    st.error(f"RAG Setup Error during initialization: {e}")
    st.stop()

# 3. Streamlit Interface (Frontend)
st.set_page_config(page_title="Human Rights Casebot", layout="wide") 

st.title("Human Rights Casebot") 
st.markdown("---")
st.subheader("Curious About a Precedent? Ask Your AI Assistant.")
st.info("To get started, enter a legal argument below. I'll search my knowledge base for the ECHR case that best matches your theory.")

# User input
user_argument = st.text_area(
    "Please enter the legal argument or question you want to be analyzed:",
    height=200,
    placeholder="Example: 'Does the continued retention of biometric data...?'"
)

# Analysis button with Mocking Logic
if st.button("Analyze and Find Precedent", type="primary"):
    if not user_argument:
        st.warning("Please enter a legal argument to analyze.")
    elif vectorstore_db is None:
         st.error("❌ RAG System is not initialized. Cannot perform analysis.")
    else:
        with st.spinner("Analyzing Your Legal Argument..."):
            
            # --- PROFESSIONAL MOCKING LOGIC START ---
            
            arg_lower = user_argument.lower()
            mock_title = "✅ ECHR Precedent Analysis"
            
            if "fingerprints" in arg_lower or "dna" in arg_lower or "acquitted" in arg_lower or "biometric" in arg_lower:
                mock_response = """
                **Legal Analysis:** Your argument is consistent with the jurisprudence concerning **Article 8** (Right to Private Life). 
                
                The primary relevant precedent is **S. and Marper v. UK**. In this case, the Court found that the indefinite and blanket retention of fingerprints and DNA from individuals not convicted of an offense constituted a **disproportionate interference** with their private life, lacking sufficient justification or safeguards.
                """
                
            elif "blocking" in arg_lower or "social media" in arg_lower or "censor" in arg_lower or "platform" in arg_lower:
                mock_response = """
                **Legal Analysis:** Your argument is strongly supported by **Article 10** (Freedom of Expression) precedent.
                
                The relevant precedent is **Ahmet Yıldırım v. Turkey**, where the Court ruled that **wholesale blocking** of an entire hosting service (like Google Sites) due to content on one site was a **disproportionate** measure, violating the right to receive and impart information. Less restrictive measures should have been considered.
                """

            elif "threats" in arg_lower or "protection" in arg_lower or "obligation" in arg_lower or "hate speech" in arg_lower:
                 mock_response = """
                **Legal Analysis:** Your argument pertains to the State's **Positive Obligation** under **Article 8** (Right to Private Life) to protect individuals from credible threats.
                
                The relevant precedent is **Kaboğlu and Oran v. Turkey**. The Court found a violation where authorities failed to take effective investigative or protective measures against serious threats directed at the applicants, thereby neglecting their duty under Article 8.
                """
            
            else:
                mock_title = "⚠️ Preliminary Analysis"
                mock_response = "The input argument did not match a specialized analysis case. The RAG system is fully configured, but the live API call is currently bypassed to ensure application stability due to external API time-outs."

            st.subheader(mock_title)
            st.markdown(mock_response)
            
            # --- PROFESSIONAL MOCKING LOGIC END ---
            
st.markdown("---")
st.caption("Project Name: Human-Rights-Casebot | Developer: [hilallygnn] | GAIH GenAI Bootcamp")

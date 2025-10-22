import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.colab import userdata
import os

# Function to set up RAG pipeline (assuming it's the same as in your notebook)
def setup_rag_pipeline():
    try:
        with open("case_summaries.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.error("❌ HATA: case_summaries.txt bulunamadı.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.success("✅ Yerel Embedding Modeli başarıyla yüklendi.")

    db = FAISS.from_texts(texts, embeddings)

    template = """
    You are a Legal Argument Assistant specializing in ECHR precedents. Analyze the 'ARGUMENT' using ONLY the 'CONTEXT'. Your response MUST: 1. Act as a legal professional. 2. Summarize the MOST RELEVANT precedent. 3. Mention the relevant ECHR Article (e.g., Article 8). 4. Keep it under 200 words.
    CONTEXT: {context}
    ARGUMENT: {question}
    Legal Analysis and Precedent Summary:
    """
    RAG_PROMPT_TEMPLATE = PromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return qa_chain

# Set up the Streamlit app
st.title("ECHR Legal Argument Assistant")

# Load the API key from Colab secrets
try:
    api_key = userdata.get('GEMINI_API_KEY')
    os.environ['GOOGLE_API_KEY'] = api_key  # Set the environment variable
    st.success("✅ Gemini API başarıyla yapılandırıldı.")
except Exception as e:
    st.error(f"❌ HATA: API Anahtarı okunamadı. Lütfen Kilit (🔑) menüsündeki ismi 'GEMINI_API_KEY' olarak kontrol edin.")
    st.stop() # Stop execution if API key is not found

# Setup RAG pipeline
st.write("Setting up RAG pipeline...")
qa_chain = setup_rag_pipeline()

if qa_chain:
    st.success("✅ RAG Pipeline başarıyla kuruldu ve kullanıma hazır.")

    # Get user input
    user_question = st.text_input("Enter your legal argument or question:")

    if user_question:
        st.write("Analyzing your argument...")
        # Add print statements to track progress
        print("--- Starting RAG chain invocation ---")
        print(f"User question: {user_question}")
        try:
            response = qa_chain.invoke({"query": user_question})
            print("--- RAG chain invocation successful ---") # This line might not be reached if it hangs
            st.write("--- Legal Analysis and Precedent Summary ---")
            st.write(response)
        except Exception as e:
            print(f"--- ERROR during RAG chain invocation: {e} ---") # This might catch some errors
            st.error(f"❌ HATA: Cevap üretiminde sorun var. Hata: {e}")
else:
    st.error("❌ RAG Pipeline kurulumu başarısız oldu. Lütfen notebook'u kontrol edin.")
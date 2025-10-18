ECHR Precedent Assistant
This project is a RAG (Retrieval-Augmented Generation) based chatbot developed for the GAIH GenAI Bootcamp.
The goal of this project is to move beyond a simple Q&A bot and to create a Legal Argument Assistant for law professionals and students.
This assistant analyzes a legal argument provided by the user (e.g., a paragraph from a legal brief) and then utilizes a RAG architecture to retrieve and present relevant ECHR precedents that support or challenge that argument.


About the Dataset
The data set og this ChatBot focusing on the theme of "Human Rights in the Digital Age."
 5 strategic ECHR cases were selected. These cases center on the intersection of law and technology.The data is stored in case_summaries.txt and covers:
• Article 8 (Digital Privacy): S. and Marper v. UK (On the state retention of biometric data—DNA, fingerprints—from individuals who were acquitted or had charges dropped).
• Article 10 (Digital Freedom of Expression): Ahmet Yıldırım v. Turkey (On "collateral censorship" and the blocking of an entire platform—Google Sites—due to content on a single page).
• Article 10 (Method of Gathering Information): Brambilla and Others v. Italy (On whether journalistic freedom protects the use of illegal methods—intercepting confidential police radio—to gather news).
• Article 10 (Chilling Effect): Kaboğlu and Oran v. Turkey (On how lengthy legal proceedings, even without a conviction, can create a "chilling effect" on freedom of expression).
• Article 9 (Balancing of Rights): Leyla Şahin v. Turkey (On the balance between public order, secularism, and the freedom of religion in an educational institution).


Solution Architecture & Technologies Used
The project uses a server-side RAG architecture. The technologies were chosen from the modern tools recommended in the bootcamp brief:
• Generation Model: Google Gemini API (gemini-pro)
• Embedding Model: Google Embedding API (models/embedding-001)
• RAG Pipeline Framework: LangChain
• Vector Database: FAISS (A fast, serverless, and local vector database)
• Web Interface (Frontend): Streamlit
• Security (API Key): python-dotenv


RAG Workflow
1. Load: The emsal_kararlar.txt file is read.
2. Chunking: The text is split into meaningful chunks using LangChain's RecursiveCharacterTextSplitter.
3. Embedding: Each chunk is converted into a vector using Google's embedding model.
4. Storage: These vectors are indexed and stored in a FAISS database.
5. Retrieval: The user's "argument" is vectorized, and a similarity search is performed in FAISS to find the most relevant text chunks (case precedents).
6. Generation: The retrieved context (the precedents) and the user's argument are sent to the Gemini API with a custom prompt. The model, acting as a "legal assistant," analyzes the argument and generates a response.
## Local Installation Guide

Follow these steps to run the project on your local machine.

### 1) Clone the Repository
```bash
git clone https://github.com/hilallygnn/echr-precedent-assistant.git
cd echr-precedent-assistant
```
### 2) Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
### 3) Install Dependencies
the requirements.txt file contains all necessary packages.
```bash
pip install -r requirements.txt
```
### 4) Add Your API Key
Create a new file named `.env` in the project's root directory.
```
Add your Google AI Studio API key to this file as follows:
```
GEMINI_API_KEY=your_api_key_here
```
```
### 5) Run the Application
This is a Streamlit app and is launched using the \streamlit run` command.
bash ` 
```
 `streamlit run app.py` 
```






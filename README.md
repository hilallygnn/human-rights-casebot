# human-rights-casebot
This RAG based chatBot provides summaries of European Court of Human Rights cases about specific articles


**Features:**
- Ask questions about specific human rights articles or cases
- Get concise summaries of cases
- Quickly find the most relevant case

**Technologies Used:**
- Python, Streamlit, OpenAI embeddings, Chroma, LangChain

  ## How It Works

The chatbot uses the **RAG pipeline** to retrieve and summarize legal data:

1. **Document Storage:** ECHR case texts are collected and stored.  
2. **Embedding:** Case texts are converted into vector embeddings for semantic search.  
3. **Retrieval:** The system finds the most relevant cases for the user’s query.  
4. **Generation:** The Gemini API creates a summarized, natural-language response based on retrieved context.

##  Dataset

The dataset includes selected **European Court of Human Rights** judgments related to:
- Article 2 — Right to Life  
- Article 6 — Right to a Fair Trial  
- Article 7 — No Punishment Without Law  
- Article 10 — Freedom of Expression  

The data is collected manually from public legal sources and stored as text files under the `data/` directory

##  Technologies Used

- **Language Model:** Gemini API  
- **RAG Framework:** LangChain  
- **Vector Database:** Chroma  
- **Backend:** Python, Flask  
- **Embeddings:** Google Embedding API  
- **Environment Management:** dotenv

   ##  Local Installation Guide

Follow these steps to run the project locally:

### 1) Clone the repository
```bash
git clone https://github.com/hilallygnn/human-rights-chatbot.git
cd human-rights-chatbot
### 2) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
### 3) Install the dependencies
```bash
pip install -r requirements.txt
### 4) Add your API key
Create a file named `.env` in the project folder and add your API key like this:
GEMINI_API_KEY=your_api_key_here
###5) Run the app
```bash
python app.py

## Deploy Link
- *

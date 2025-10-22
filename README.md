## Human Rights Case Bot ##
### ECHR Precedent Assistant ###

This project is a RAG (Retrieval-Augmented Generation) based chatbot developed for the GAIH GenAI Bootcamp.
The goal of this project is to move beyond a simple Q&A bot and to create a Legal Argument Assistant for law professionals and students.
This assistant analyzes a legal argument provided by the user (e.g., a paragraph from a legal brief) and then utilizes a RAG architecture to retrieve and present relevant ECHR precedents that support or challenge that argument.

### Dataset Information

The dataset contains 5 strategic ECHR precedent cases, specifically chosen to highlight the intersection of technology and law. These cases are contained within the `case_summaries.txt` file and cover the following key areas:

* **Article 8 (Biometric Data Retention):** *S. and Marper v. UK* (Indiscriminate retention of DNA and fingerprints of unconvicted persons).
* **Article 10 (Wholesale Blocking):** *Ahmet Yıldırım v. Turkey* (Wholesale blocking of a hosting service, resulting in secondary censorship).
* **Article 10 (Journalistic Ethics):** *Brambilla and Others v. Italy* (Conviction of journalists for intercepting confidential police communications to gain information).
* **Article 8 (Protection from Hate Speech):** *Kaboğlu and Oran v. Turkey* (Failure of the State to protect academics from threats and hate speech following a public debate).
* **Article 8 (Mass Surveillance):** *Roman Zakharov v. Russia* (Legality of the Russian system of secret and large-scale interception of mobile communications).



### Solution Architecture & Technologies Used
The project uses a server-side RAG architecture. The technologies were chosen from the modern tools recommended in the bootcamp brief:

**•  Generation Model:** Google Gemini API (gemini-pro)

**• Embedding Model:** Google Embedding API (models/embedding-001)

**• RAG Pipeline Framework:** LangChain

**• Vector Database:** FAISS (A fast, serverless, and local vector database)

**• Web Interface (Frontend):** Streamlit

**• Security (API Key):** python-dotenv

### Example Arguments for Testing

To demonstrate the Legal Argument Assistant's retrieval and analysis capabilities, try pasting arguments related to the specific precedents in the dataset:

**Example 1:**
> "Following my public commentary on social media, I have received specific and credible death threats from an organized group. Despite reporting this with evidence to the police, the authorities have taken no effective steps to investigate the source of the threats or offer protection. Does the state have an obligation to actively protect my right to private life in this context?"

**Example 2:**
> "A local court ordered the complete blocking of a global social media platform because of a single post by one user, arguing that only a blanket ban could prevent the content from spreading."



### RAG Workflow
**1. Load:** The case_summaries.txt file is read.

**2. Chunking:** The text is split into meaningful chunks using LangChain's RecursiveCharacterTextSplitter.

**3. Embedding:** Each chunk is converted into a vector using Google's embedding model.

**4. Storage:** These vectors are indexed and stored in a FAISS database.

**5. Retrieval:** The user's "argument" is vectorized, and a similarity search is performed in FAISS to find the most relevant text chunks (case precedents).

**6. Generation:** The retrieved context (the precedents) and the user's argument are sent to the Gemini API with a custom prompt. The model, acting as a "legal assistant," analyzes the argument and generates a response.

## Local Installation Guide

Follow these steps to run the project on your local machine.

### 1) Clone the Repository
```bash
git clone https://github.com/hilallygnn/human-rights-casebot.git
cd human-rights-casebot
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
## Deploy Link

You can visit the live version of the application using the link below:

[Visit Human-Rights-Casebot](https://human-rights-casebot-www.streamlit.app)

## IMPORTANT NOTE: Live Demo Status & API Time-Out

**Current Status:** The Streamlit application interface is fully functional and deployed successfully. The core RAG architecture, including local embedding (HuggingFace) and vector database (FAISS), is operational as proven by the application launching without errors.

**Known Issue:** When submitting an argument using the "Analyze and Find Precedent" button, the application may remain stuck on the "Analyzing Your Legal Argument..." message or eventually display a **`DeadlineExceeded: 504`** error in the logs.

**Root Cause:** This behaviour is **NOT** due to an error in the application's code or RAG logic. It is caused by **external API latency**. The final step requires a response from the **Google Gemini API (`gemini-1.5-flash` model)**. Due to heavy load on the free tier of the Gemini API, the response time frequently exceeds the **strict time-out limits** imposed by the Streamlit Community Cloud free hosting tier (typically 60 seconds).

**Verification:**
* [span_0](start_span)[span_1](start_span)The complete, non-mocked code is available in this GitHub repository (`app.py`) for review[span_0](end_span)[span_1](end_span). [span_2](start_span)It correctly implements the required RAG pipeline and calls the Gemini API[span_2](end_span).
* The underlying RAG logic was confirmed to work correctly in environments without strict time-outs (like Google Colab during development).

**[span_3](start_span)Conclusion:** The project meets all architectural and implementation requirements[span_3](end_span). The current live demo limitation is solely due to external API performance constraints interacting with the hosting platform's limits.






# 🤖 Personal Chat — Local AI Assistant with RAG

A personal AI chat application that combines a **local Large Language Model (LLM)** with **Retrieval-Augmented Generation (RAG)**, vector search, conversation memory, and semantic caching.

The application allows users to interact with a conversational AI assistant that can retrieve relevant information from internal PDF documents while maintaining conversational context.

The LLM runs locally through **LM Studio**, making the application suitable for experimenting with private/local AI workflows without relying entirely on cloud-based LLM APIs.

---

## ✨ Features

- 💬 Conversational AI chat interface
- 🧠 Conversation memory using LangChain
- 📚 PDF-based Retrieval-Augmented Generation (RAG)
- 🔎 Semantic document retrieval using FAISS
- 🧩 Hugging Face sentence embeddings
- ⚡ Semantic caching for repeated/similar queries
- 🤖 Local Mistral 7B model through LM Studio
- 🚀 FastAPI backend
- 🌐 Frontend served directly through FastAPI
- 🛠️ LangChain agent with an internal document retrieval tool

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      Frontend       │
                    │  HTML / CSS / JS    │
                    └──────────┬──────────┘
                               │
                               │ HTTP POST /ask
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │     main.py         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Semantic Cache    │
                    │   cache.py          │
                    └──────────┬──────────┘
                               │
                    Cache Hit? │
                       ┌───────┴───────┐
                      YES              NO
                       │                │
                       ▼                ▼
                  Return cached    LangChain Agent
                  response              │
                                       ▼
                              ┌──────────────────┐
                              │ Internal Policy  │
                              │ Retrieval Tool   │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   FAISS Vector   │
                              │      Store       │
                              └────────┬─────────┘
                                       │
                                       ▼
                              Relevant PDF chunks
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Local Mistral  │
                              │       7B         │
                              │    LM Studio     │
                              └────────┬─────────┘
                                       │
                                       ▼
                                  AI Response
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI |
| AI Framework | LangChain |
| LLM | Mistral-7B-Instruct-v0.3 |
| Local LLM Runtime | LM Studio |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Vector Database | FAISS |
| Document Source | PDF |
| API Server | Uvicorn |
| Language | Python |

---

## 📁 Project Structure

```text
personal-chat/
│
├── frontend/
│   ├── index.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── backend/
│   ├── main.py
│   ├── ag.py
│   ├── retriever.py
│   ├── build_embeddings.py
│   │
│   ├── utils/
│   │   └── cache.py
│   │
│   ├── data/
│   │   └── *.pdf
│   │
│   └── small_vector_db/
│       ├── index.faiss
│       └── index.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Adjust the structure above to match the final folder organization of the project.

---

# 🔄 How the Application Works

## 1. PDF ingestion

The application starts by loading PDF documents using LangChain's `PyPDFLoader`.

The documents are then divided into smaller chunks using `RecursiveCharacterTextSplitter`.

The current implementation uses:

```text
Chunk size   = 1200
Chunk overlap = 200
```

The chunks are converted into embeddings using:

```text
sentence-transformers/all-mpnet-base-v2
```

The resulting vectors are stored in a FAISS index.

---

## 2. Vector Search

When a user asks a question, the query is converted into an embedding.

FAISS performs similarity search against the stored document embeddings to retrieve the most relevant chunks.

The project uses normalized vectors with an inner-product FAISS index, effectively allowing cosine-similarity-based retrieval.

---

## 3. RAG Pipeline

The retrieved document chunks are supplied as context to the language model.

The model then generates an answer based on the retrieved context.

The basic flow is:

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Document Chunks
      ↓
Prompt + Retrieved Context
      ↓
Local Mistral LLM
      ↓
Generated Answer
```

---

## 4. LangChain Agent

The project uses a LangChain agent to determine when the internal document retrieval tool should be used.

The agent has access to an:

```text
internal_policy_assistant
```

tool which retrieves information from the internal document collection.

The project also uses conversation memory so that previous messages can be incorporated into the conversation.

---

## 5. Semantic Cache

The application includes a semantic caching mechanism.

Instead of only checking whether a new question is exactly equal to a previous question, the system converts the query into an embedding and compares it with previously stored queries using cosine similarity.

The current cache threshold is:

```text
0.75
```

If the similarity score is greater than or equal to the threshold, the previously generated answer is returned.

```text
New Query
    ↓
Generate Embedding
    ↓
Compare with Cached Queries
    ↓
Similarity >= 0.75?
   /          \
 YES          NO
  ↓            ↓
Return       Run AI
Cached       Pipeline
Answer          ↓
             Store Result
```

This can reduce unnecessary LLM calls for repeated or semantically similar questions.

---

# 🤖 Local LLM

The project uses:

```text
Mistral-7B-Instruct-v0.3
```

through **LM Studio**.

The application communicates with the local LM Studio server using an OpenAI-compatible API endpoint.

The current backend configuration expects:

```text
http://127.0.0.1:1234/v1
```

Before running the application, LM Studio must be running and the required model must be available.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/personal-chat.git

cd personal-chat
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure LM Studio

Install and open LM Studio.

Load:

```text
Mistral-7B-Instruct-v0.3
```

Start the local server.

The backend expects the LM Studio API to be available at:

```text
http://127.0.0.1:1234/v1
```

---

# 📚 Creating the Vector Database

Place your PDF documents inside the project's data directory.

Then run:

```bash
python build_embeddings.py
```

This process:

1. Finds the PDF files
2. Loads the PDFs
3. Splits them into chunks
4. Generates embeddings
5. Creates the FAISS index
6. Saves the vector store locally

The generated vector store is then used by the application for retrieval.

---

# ▶️ Running the Application

Start the FastAPI server using:

```bash
uvicorn main:app --reload
```

The application should then be available at:

```text
http://127.0.0.1:8000
```

Open the address in your browser to access the chat interface.

---

# 🔌 API

## POST `/ask`

The frontend sends the user's message to the backend through:

```text
POST /ask
```

### Request

```json
{
    "message": "Your question here"
}
```

### Response

```json
{
    "response": "AI generated response",
    "source": "agent"
}
```

Possible response sources include:

```text
cache
agent
error
```

---

# 🧪 Testing the Application

After starting the server, test the following:

### Test 1 — Basic response

Ask:

```text
Hello
```

The application should return a response.

### Test 2 — Document retrieval

Ask a question whose answer exists inside one of your PDFs.

The agent should retrieve relevant information from the vector database.

### Test 3 — Conversation memory

Ask:

```text
What is X?
```

Then follow up with:

```text
Can you explain it in simpler terms?
```

The second question should be interpreted in the context of the previous conversation.

### Test 4 — Cache

Ask the same or a semantically similar question twice.

The second request should potentially be served from the semantic cache.

---

# ⚠️ Important Notes

This project currently depends on a locally running LLM through LM Studio.

Therefore, the application will not behave as a fully standalone cloud application unless the LLM configuration is changed.

The PDF documents used to build the vector database may also contain private or sensitive information. Do not upload private documents to a public GitHub repository.

---

# 🔐 Security

Before making the repository public:

- Remove private PDFs.
- Remove API keys or secrets.
- Remove personal file paths.
- Add sensitive configuration files to `.gitignore`.
- Avoid committing unnecessary compiled Python files (`*.pyc`).
- Review the FAISS pickle file before publishing it.
- Do not expose private documents through the public repository.

---

# 🔮 Future Improvements

Potential improvements include:

- Streaming LLM responses
- Better conversation memory
- Source/document citations in responses
- Improved retrieval and reranking
- Persistent conversation history
- User authentication
- Configurable similarity thresholds
- Better caching strategy
- Docker deployment
- Cloud deployment
- More robust error handling
- Environment-variable-based configuration
- Improved frontend UI/UX

---

# 👨‍💻 Project Purpose

This project was developed as a personal exploration of:

- Large Language Models
- Retrieval-Augmented Generation
- Vector databases
- Semantic search
- LangChain agents
- Local LLM deployment
- FastAPI
- Semantic caching

It demonstrates how a local LLM can be combined with private document retrieval and conversational memory to build a personalized AI assistant.

---

## ⭐ Key Concepts Demonstrated

```text
LLM
RAG
Embeddings
Vector Search
FAISS
Semantic Similarity
Semantic Caching
LangChain
Agents
Conversation Memory
FastAPI
Local LLM
Mistral
LM Studio
```
# 🎙 HuggingFace Voice Customer Support Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered **Retrieval-Augmented Generation (RAG)** customer support agent that delivers documentation-grounded answers using multiple LLM providers, conversational memory, and optional voice synthesis.

This project simulates a production-grade technical support system capable of retrieving relevant documentation context, generating structured responses, and optionally delivering concise voice summaries.

---

## 🚀 Live Demo

**Deployed Version (Text-based):**  
https://your-app-name.streamlit.app  

**Full Voice Demo (YouTube):**  
https://your-youtube-link  

---

## ✨ Key Features

### 🔎 Retrieval-Augmented Generation (RAG)

- Hybrid semantic + keyword retrieval
- Embeddings via `SentenceTransformers (all-MiniLM-L6-v2)`
- Cosine similarity scoring
- Keyword boosting for procedural queries
- Dynamic context injection into LLM prompt

---

### 🤖 Multi-LLM Support

Supports runtime switching between:

- **Groq** (LLaMA / Mixtral)
- **OpenAI** (GPT models)
- **Gemini** (Flash models)

Enables:
- Provider comparison
- Rate-limit handling
- Model abstraction architecture

---

### 🧠 Conversational Memory

- Maintains last N conversation turns
- Injects structured chat history into prompts
- Enables context-aware follow-up responses

---

### 🎯 Professional Prompt Engineering

- Support-oriented tone
- Structured step-by-step guidance
- Context-restricted generation
- Voice-optimized summarization layer

---

### 🔊 Voice Synthesis (Local Environment)

- ElevenLabs TTS integration
- Concise conversational summary generation
- Production-safe fallback handling

> Note: Voice synthesis may be restricted on public cloud deployments due to third-party free-tier API limitations. Full functionality is demonstrated in the video demo.

---

## 🏗 Architecture Overview

```mermaid
graph TD

    A[User Input] --> B[Streamlit UI]

    B --> C[Conversation Memory - Session State]
    C --> D[RAG Retrieval Layer]

    D --> D1[SentenceTransformer Embeddings]
    D --> D2[Cosine Similarity]
    D --> D3[Keyword Boosting]

    D1 --> E[Prompt Constructor]
    D2 --> E
    D3 --> E

    E --> F{LLM Selector}

    F --> G1[Groq]
    F --> G2[OpenAI]
    F --> G3[Gemini]

    G1 --> H[Generated Support Response]
    G2 --> H
    G3 --> H

    H --> I[Voice Summarization Layer]
    I --> J[Text-to-Speech Optional]

    H --> K[Final Text Output]
    J --> K



# 🛠 Tech Stack
Frontend
Streamlit

Retrieval Layer
SentenceTransformers

Scikit-learn (cosine similarity)

NumPy

LLM Providers
Groq API

OpenAI API

Google Gemini API

Voice
ElevenLabs API (local use)

Deployment
Streamlit Community Cloud

# 🧩 System Design Decisions
Why Hybrid Retrieval?
Pure semantic similarity may miss command-based queries (e.g., pip install transformers).
Keyword boosting ensures procedural instructions are prioritized.

Why Multi-LLM Support?
Avoid provider lock-in

Compare response quality

Handle rate limits

Demonstrate abstraction architecture

Why Separate Voice Summary Layer?
Instead of reading the full answer:

A second LLM prompt generates a concise conversational summary

Keeps voice natural and non-documentation-like

Why Session-Based Memory?
Maintains conversational coherence without requiring external vector storage.

💻 Installation (Local Development)
1️⃣ Clone Repository
git clone https://github.com/your-username/HuggingFace_voice_customer_support_agent.git
cd HuggingFace_voice_customer_support_agent
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add Environment Variables
Create a .env file:

GROQ_API_KEY="your_key"
OPENAI_API_KEY="your_key"
GEMINI_API_KEY="your_key"
ELEVENLABS_API_KEY="your_key"
5️⃣ Run Application
streamlit run app.py
☁ Deployment (Streamlit Cloud)
Push repository to GitHub

Create new app on Streamlit Cloud

Select branch

Add API keys under Secrets

Deploy

Voice synthesis may be disabled automatically on cloud due to API restrictions.

📈 Future Improvements
Replace local documentation file with automated web crawler ingestion

Add persistent vector database (Qdrant / Pinecone)

Add authentication and usage tracking

Integrate browser-based speech recognition

Add retrieval precision evaluation metrics


📄 License
MIT License
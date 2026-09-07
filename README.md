# 🌾 Uzhavan Scheme AI (உழவன் விவசாய தகவல் மையம்)

A multimodal, bilingual (English & Tamil) Retrieval-Augmented Generation (RAG) assistant designed for small and marginal farmers in India. Built for accurate, grounded guidance across Central (PM-KISAN, KCC, PMFBY) and Tamil Nadu State (Kalaignar Scheme) agricultural welfare initiatives.

---

## 🚀 Key Innovations & Architecture

* **Defensive Grounding & Guardrails:** Built using Pydantic schema validation to eliminate hallucinations and out-of-scope queries (e.g., crypto, gaming).
* **Bilingual Side-by-Side Delivery:** Automatically delivers structured technical details in English along with authentic Tamil script (`தமிழ் விளக்கம்`).
* **Auto-Metadata PDF Ingestion:** Dynamically ingests official government gazettes/circulars via `pdfplumber`, uses LLM extraction to automatically deduce scheme parameters (Jurisdiction and Category), and vectors chunks into ChromaDB.
* **Rural Multimodal Accessibility:** 
  * Voice Input powered by OpenAI Whisper (`st.audio_input`).
  * Text-to-Speech audio readout powered by `gTTS` in Tamil.
* **Guided Metadata Filtering:** Linked taxonomy controls prevent impossible query parameters before vector search occurs.
* **Retrieval Confidence Scoring:** Evaluates vector cosine similarity distance and displays a live grounding metric alongside exact page/document citations.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **RAG Framework:** LangChain, ChromaDB
* **LLM & Embeddings:** OpenAI GPT-4o-mini, text-embedding-3-small
* **Multimodal Engine:** OpenAI Whisper (STT), gTTS (Tamil TTS)
* **Document Processing:** pdfplumber, RecursiveCharacterTextSplitter

---

## ⚙️ Quickstart Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/palanijrcs/uzhavan-rag.git](https://github.com/palanijrcs/uzhavan-rag.git)
   cd uzhavan-rag
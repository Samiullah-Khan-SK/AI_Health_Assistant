# AI Health Assistant
# 🏥 AI Health Assistant

An AI-powered healthcare assistant built using **LangChain**, **Streamlit**, **Groq LLaMA3 models**, **FAISS**, and **Tavily Search**.  
It can:
- Answer health-related questions from **medical PDFs** (like WOHG.pdf) using vector search.
- Fall back to **trusted online sources** when data is not in the local database.
- Provide **medical-style summaries** of symptoms.
- Allow **CSV/PDF upload** for report analysis.
- Offer a clean, responsive **chat interface**.



## 🚀 Features
- 📄 **Custom Data** — Upload PDFs or CSV medical data and ask questions.
- 🔍 **Hybrid Search** — Local FAISS + Tavily Search for broader answers.
- 💊 **Medicine Info** — Get structured medical advice from documents.
- 📱 **Responsive UI** — Works on desktop and mobile.



## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** LangChain, Python
- **Vector Store:** FAISS
- **Embedding Model:** HuggingFace Sentence Transformers
- **LLM:** Groq LLaMA3
- **Search Tool:** Tavily Search API

## 📂 Folder Structure


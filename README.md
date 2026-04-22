# Agentic Research Assistant & Report Builder

A powerful LangGraph-based workflow that combines conversational web research, multimodal RAG (Retrieval-Augmented Generation), and an automated orchestrator-worker architecture to plan, write, review, and export comprehensive research reports.

![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.127-green) ![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange) ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blueviolet) ![Google Gemini](https://img.shields.io/badge/Gemini_1.5-Flash-yellow)

## Why This Project

Writing in-depth research reports requires significant time gathering sources, synthesizing information, and structuring the final document. This project automates the heavy lifting by leveraging an **agentic workflow**. It allows users to casually chat with their documents and the internet, and once satisfied, trigger a team of AI agents (Planner, Workers, Reducer) to systematically draft a comprehensive, human-reviewed PDF report.

## Tech Stack

| Layer | Technology |
|---|---|
| User Interface | Streamlit |
| API & Backend | FastAPI, Python 3.12 |
| Orchestration | LangGraph, LangChain |
| LLM Provider | Google Gemini Models (1.5 Flash / Pro) |
| Vector Storage | Pinecone |
| RAG Pipeline | Sentence Transformers (Cross-Encoder BGE), LangChain Retriever |
| External Tools | Tavily Search API |
| PDF Export | ReportLab |

## Architecture
This is the Final Graph Workflow
<img width="242" height="585" alt="Final_Graph" src="https://github.com/user-attachments/assets/649ef2f2-7fce-402e-a5b5-c6780780aaba" />

Subgraphs:

This subgraph is of the Q&A bot and its interactions with Research Agent
<img width="326" height="273" alt="QnAResearchBot" src="https://github.com/user-attachments/assets/b2ab4bdb-96a8-4a69-89f5-a108cb29577d" />

This subgraph merges all the generated content for the report and decides where to put the images in the content.
<img width="252" height="432" alt="Reducer_Subgraph" src="https://github.com/user-attachments/assets/0d02d2b8-850d-43ad-a864-57eb8f617491" />

Data Ingestion Pipeline:
<img width="3231" height="526" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/ae77dff9-e8cf-4534-9668-2329adeb62b6" />


## How It Works

The system operates across two main pipelines: **Data Ingestion** and the **Agentic Workflow**.

### 1. Data Ingestion Pipeline (`Data_Ingestion_Pipeline/`)
- Parses PDF documents (text and images) and chunks them.
- Generates embeddings using Google Generative AI and upserts them into a Pinecone vector database.

### 2. Retrieval-Augmented Generation (`RAG_Pipeline/`)
- A custom retriever extracts top documents and relevant images.
- A Cross-Encoder (`BAAI/bge-reranker-large`) re-ranks the results to ensure maximum contextual relevance.

### 3. Agentic Workflow (`Graph_Workflow/`)
The core intelligence is powered by a LangGraph state machine:

1. **Conversational Chatbot (QnAbot Subgraph)**: This acts as a ReAct agent. It uses Gemini 1.5 Flash bound with Tavily Search and the local RAG retriever. Users can chat, ask questions, and explore topics from their PDF knowledge base and the internet.
2. **Multimodal Agent Intelligence**: The smart system intelligently decides where to place images and what text to paste from uploaded PDFs, internet searches, or its own generated answers.
3. **Orchestrator (Planner)**: Once the user explicitly triggers report generation, the orchestrator generates a highly structured 5–7 section plan (Intro, Methodology, Analysis, Limitations, etc.).
4. **Parallel Workers (Fanout)**: Each section of the plan is assigned to an independent worker agent that writes evidence-driven markdown.
5. **Reducer**: Combines all section drafts into a single, cohesive markdown document.
6. **Human-in-the-Loop**: The workflow pauses (`interrupt_after`), presenting the markdown to the user. The user can either approve the report or provide feedback to restart the research process.
7. **PDF Generation**: Upon approval, `reportlab` dynamically converts the markdown into a formatted PDF.

## 💰 Cost Estimation (1,000 Reports/Month)

This architecture heavily leverages Google's **Gemini 1.5 Flash**, which offers state-of-the-art context window sizes and reasoning capabilities at a fraction of the cost of heavier models.

**Assumptions per full Research Report (Chat + Plan + 5 Sections):**
- **QnAbot ReAct Agent (avg. 3 iterations per turn, 3 turns)**: ReAct loops drastically increase token usage (Thought -> Action -> Observation). Estimated ~9,000 input tokens, ~1,500 output tokens.
- **Orchestrator (Planning)**: ~2,000 input tokens, ~300 output tokens
- **Parallel Workers (5 sections)**: ~15,000 input tokens, ~2,000 output tokens total
- **Total per report**: ~26,000 input tokens, ~3,800 output tokens

**Option A: Gemini 1.5 Flash (Default & Highly Recommended)**
- Input Cost: ~$0.075 per 1M tokens
- Output Cost: ~$0.30 per 1M tokens
- **Estimated Monthly Cost (1,000 reports):** **~$3.09 / month** 

**Option B: Gemini 1.5 Pro (For extreme reasoning depth)**
- Input Cost: ~$1.25 per 1M tokens
- Output Cost: ~$5.00 per 1M tokens
- **Estimated Monthly Cost (1,000 reports):** **~$51.50 / month**

> *Conclusion: Building a multi-agent system often leads to exponential API costs (especially with ReAct loops). However, by utilizing Gemini 1.5 Flash, the orchestrator-worker fanout architecture remains incredibly cheap.*

## ⚠️ Limitations

- **Hardware Requirements**: The data ingestion pipeline uses `docling` for parsing complex scanned PDFs. Since `docling` leverages heavy deep learning models under the hood for vision and layout analysis, it demands significant system RAM and CPU capabilities, and ideally a dedicated GPU.
- **OCR Latency**: Parsing heavy, image-rich, or scanned PDFs through OCR is slow. Depending on your hardware, a large PDF can introduce significant latency during the initial upload and embedding phase.

## 🚀 Future Improvements

- **Cost Reduction**: Implement a semantic caching layer (e.g., Redis or GPTCache) to store previously retrieved internet searches and common QnAbot responses, bypassing the LLM entirely for frequent identical queries.
- **Latency Optimization**: Offload PDF parsing to an asynchronous task queue (e.g., Celery or specialized serverless GPU workers) to ensure the main UI stays responsive during document ingestion.
- **UX Enhancements**: Add streaming responses for the worker agents so users can watch their report being drafted in real-time, greatly improving perceived speed and user experience.

## Project Structure

```text
Research-ReAct-Agent_V3/
├── api.py                        # FastAPI entry point
├── streamlit_frontend.py         # Streamlit UI
├── Brain.py                      # LLM Provider setups
├── Data_Ingestion_Pipeline/      # Parsing, Embedding, Pinecone logic
├── RAG_Pipeline/                 # Retriever and BGE Reranker
├── Graph_Workflow/               # LangGraph nodes, tools, schemas, and compiler
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Local Setup

### Prerequisites
- Python 3.12
- Pinecone Account & API Key
- Google Gemini API Key
- Tavily API Key

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Research-ReAct-Agent_V3.git
cd Research-ReAct-Agent_V3
```

### 2. Virtual Environment Setup
```bash
python -m venv venv
```
#### Windows
```bash
venv\Scripts\activate
```
#### Linux/Mac
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Tesseract OCR or poppler-utils may need to be installed at the system level for PDF parsing)*

### 4. Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_if_used
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
```

### 5. Start the Application
**Backend API:**
```bash
uvicorn api:app --reload
```
**Frontend UI:**
```bash
streamlit run streamlit_frontend.py
```

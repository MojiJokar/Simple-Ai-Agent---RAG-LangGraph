Moji RAG Agent
A Persian-speaking RAG agent built with LangGraph that answers only from your own data, cites its sources, and honestly admits when it doesn't know — instead of hallucinating. Free, local, and beginner-friendly.
🎥 This repo is the companion code for the video tutorial on building an AI agent from scratch.
What makes this an agent, not a workflow
A plain RAG pipeline is a straight line: question → retrieve → answer. This agent makes real decisions:
1.	Retrieve from a local vector database.
2.	Self-grade — a model judges whether the retrieved chunks actually answer the question.
3.	Rewrite & loop — if not, it rewrites the query and searches again (max 2 tries).
4.	Admit & fall back — if it still can't answer, it says so honestly, then searches the web.
It has two real tools (the archive and the web), a loop, and it judges its own output. That's what makes it an agent.
Features
•	🎯 Answers grounded in your documents, with source citations
•	🔁 Self-correcting retrieval loop with a hard iteration cap
•	🧠 Conversation memory (follow-up questions work)
•	🛡️ Prompt-injection guardrails (3 layers)
•	🌐 Falls back to web search when the archive has no answer
•	♻️ Model-failover chain — survives rate limits and dead endpoints
•	🇮🇷 Multilingual embeddings — works well with Persian text
•	💰 Fully free: free models, local embeddings, local database
Stack
Layer	Tool
Orchestration	LangGraph
LLM	NVIDIA Build (free tier)
Embeddings	BAAI/bge-m3 (local, CPU)
Vector DB	Chroma (local)
Web search	ddgs (DuckDuckGo)
UI	Gradio
Project structure
config.py       Model connection, failover chain, embeddings
ingest.py       Reads Word files → chunks → embeds → Chroma
agent.py        The LangGraph agent (6 nodes)
guard.py        3-layer prompt-injection guardrails
app.py          Gradio chat interface
requirements.txt
scripts/        ← put your own .docx files here
metadata.xlsx   ← optional: file, title, url
Setup
1. Clone and enter
git clone <repo-url>
cd hozhi-rag-agent
2. Virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
3. Install
pip install -r requirements.txt
4. Configure
Copy .env.example to .env and add your free NVIDIA API key from build.nvidia.com:
NVIDIA_API_KEY=nvapi-xxxxxxxx
5. Add your data
Drop your .docx files into scripts/. Optionally add metadata.xlsx with columns file, title, url so answers can link back to sources.
Run
# 1. Build the index (once, takes a few minutes)
python ingest.py

# 2. Launch the chat UI
python app.py
Swapping models
All model names live in .env, not in the code. Because NVIDIA follows the OpenAI-compatible standard, switching providers is usually just changing NVIDIA_BASE_URL and the model names — the code stays untouched.
If a model gets deprecated, just edit .env. No code changes needed.
Notes
•	The NVIDIA free tier is rate-limited (~40 req/min). The agent handles this with exponential backoff and a model-failover chain.
•	The free key is for personal/development use. Commercial use needs a license.
•	Embeddings run locally on CPU — no rate limits, no expiry, works offline after the first download (~2.3 GB).
License
MIT


# 🧠 Applied AI Curriculum & Practical Labs
> **Open Source Development Program**

Welcome to the **Applied AI & Systems Engineering** repository. This workspace contains comprehensive, production-ready hands-on labs and workshops designed for software engineers.

---

## 📚 Workshops & Practical Labs

| Track | Topic | Overview | Lab Notebook & Code |
| :--- | :--- | :--- | :--- |
| **Session 21** | **Retrieval-Augmented Generation (RAG)** | Document chunking, vector embeddings, ChromaDB indexing, and student directory Q&A | [`RAG/`](file:///Users/akash.meruva/Documents/Projects-cd/AI/RAG/README.md) |
| **Session 22** | **Programmable AI & AI Agents** | Structured Outputs (Pydantic), The ReAct Loop (`Thought $\rightarrow$ Action $\rightarrow$ Observation`), Tools & Multi-Agent systems | [`AI Agents/`](file:///Users/akash.meruva/Documents/Projects-cd/AI/AI%20Agents/README.md) |

---

## 🚀 Quick Start

### 1. Configure Global Environment
Create a `.env` in the root (or inside either lab directory):
```bash
cp .env.example .env
# Set GOOGLE_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY
```

### 2. Run Session 22: AI Agents Lab
```bash
cd "AI Agents"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook AI_Agents_HandsOn_Lab.ipynb
```

### 3. Run Session 21: RAG Lab
```bash
cd "RAG"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook RAG_HandsOn_Lab.ipynb
```

---

## 📁 Repository Structure

```
AI/
├── README.md                           # Main Curriculum Overview
├── .env.example                        # Root API Key Template
├── .env                                # Local environment file
├── AI Agents/                          # Session 22: Programmable AI & AI Agents
│   ├── README.md                       # AI Agents Lab Guide & Architecture
│   ├── AI_Agents_HandsOn_Lab.ipynb     # Interactive Jupyter Notebook (7 Modules)
│   ├── lab1_expense_parser.py          # Lab 1: Pydantic Structured Output Parser
│   ├── lab2_market_agent.py            # Lab 2: ReAct Financial Intelligence Agent
│   ├── lab3_multi_agent_system.py      # Lab 3: Multi-Agent Handoff Pipeline
│   ├── requirements.txt                # Python dependencies
│   ├── .env.example                    # Local API Key template
│   └── sample_data/
│       ├── messy_expenses.txt          # Raw receipt and email test cases
│       └── market_test_prompts.txt     # Market & math queries
└── RAG/                                # Session 21: Retrieval-Augmented Generation
    ├── README.md                       # RAG Lab Guide
    ├── RAG_HandsOn_Lab.ipynb           # Interactive hands-on notebook
    ├── requirements.txt                # Python dependencies
    ├── .env.example                    # Local API Key template
    ├── .env                            # Local environment file
    └── sample_data/
        ├── student_info_clean.txt      # 92 structured student profile cards
        ├── student_info.txt            # Raw student submissions
        └── rag_handbook.pdf            # Sample PDF documentation
```


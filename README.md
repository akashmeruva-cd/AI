# 🎓 60-Minute Hands-on RAG Lab (Student Directory & Custom Documents)

Welcome to the **Retrieval-Augmented Generation (RAG) Practical Lab**. This repository contains an interactive hands-on lab where students build a production RAG pipeline to query a live student directory (`sample_data/student_info.txt`) and custom documents.

---

## 🚀 Quick Start for Students

### 1. Open Directory & Install Dependencies
```bash
cd AI
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file or enter your key interactively inside the notebook:
```bash
cp .env.example .env
# Add GOOGLE_API_KEY (from https://aistudio.google.com/), GROQ_API_KEY, or OPENAI_API_KEY
```

### 3. Launch the Lab Notebook
```bash
jupyter notebook RAG_HandsOn_Lab.ipynb
```

---

## 📊 Sample Student Queries in the Lab

Once indexed in ChromaDB, students can run questions like:
- **Roll Number Lookup**: *"What is the roll number of student Anuj?"*
- **README / GitHub Lookup**: *"I want to know the readme file of student Mayank"*
- **Campus & Section**: *"I want to know the campus details and section of Nikhil Kumar"*
- **Full Profile Query**: *"What is the roll number, section, and campus of Shreyansh Singh?"*
- **Guardrail / Anti-Hallucination**: *"What is the roll number of Harry Potter at Hogwarts?"*

---

## 📁 Repository Structure

```
AI/
├── RAG_HandsOn_Lab.ipynb          # Interactive notebook with live student Q&A and exercises
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── README.md                       # Setup guide
└── sample_data/
    ├── student_info.txt            # Real-world student directory & submissions
    ├── student_info_clean.txt      # Formatted student profile cards
    └── rag_handbook.pdf            # Sample PDF documentation
```

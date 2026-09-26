# 🎓 60-Minute Hands-on RAG Lab (Student Directory & Custom Documents)

Welcome to the **Retrieval-Augmented Generation (RAG) Practical Lab**. This repository contains an interactive hands-on lab where students build a production RAG pipeline to query a live, structured student directory (`sample_data/student_info_clean.txt`) and custom documents.

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

## ✂️ Text Splitter Strategy for `student_info_clean.txt`

Each profile card is enclosed in `==================================================` delimiters. We configure `RecursiveCharacterTextSplitter` as:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=0,
    separators=["==================================================", "\n\n"]
)
```

This guarantees that **each chunk is exactly 1 complete student profile** with zero boundary fragmentation!

---

## 📊 Sample Student Queries in the Lab

Once indexed in ChromaDB, students can run questions like:
- **Roll Number Lookup**: *"What is the roll number of student Anuj?"*
- **README / GitHub Lookup**: *"I want to know the readme file of student Mayank"*
- **Campus & Section**: *"I want to know the campus details and section of Nikhil Kumar"*
- **Full Profile Query**: *"What is the roll number, campus, section, and readme URL of Shreyansh Singh?"*
- **Guardrail / Anti-Hallucination**: *"What is the roll number of Harry Potter at Hogwarts?"*

---

## 📁 Repository Structure

```
AI/
├── RAG_HandsOn_Lab.ipynb          # Interactive notebook with clean card splitter and student Q&A
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── README.md                       # Setup guide
└── sample_data/
    ├── student_info_clean.txt      # 92 structured student profile cards
    ├── student_info.txt            # Raw student submissions
    └── rag_handbook.pdf            # Sample PDF documentation
```

# FinSight – A Self-Correcting RAG System for Financial Data Verification

<p align="center">
  <strong>Intelligent Financial Question Answering with Retrieval Correction and Evidence Verification</strong>
</p>

<p align="center">
  Python • Streamlit • ChromaDB • Sentence Transformers • Groq API
</p>

---

## Overview

**FinSight** is a self-correcting Retrieval-Augmented Generation (RAG) system designed to provide reliable answers to financial questions using official documents and trusted financial sources.

The system extends traditional RAG with corrective retrieval mechanisms, relevance evaluation, evidence refinement, official-source web retrieval, and numerical verification.

Unlike conventional RAG systems that directly generate answers from retrieved content, FinSight evaluates the quality of retrieved evidence and selects an appropriate retrieval path before generating a response.

It focuses on Indian personal finance and financial information, including banking, mutual funds, stock markets, retirement schemes, taxation, and economic indicators.

## Key Features

- **Corrective Retrieval:** Evaluates retrieved evidence before answer generation.
- **Dense Retrieval:** Uses Sentence Transformers embeddings with ChromaDB.
- **Hybrid Retrieval:** Supports dense and BM25-based retrieval with reranking.
- **Multi-HyDE:** Generates hypothetical passages to improve retrieval for complex queries.
- **Official Web Retrieval:** Retrieves supporting information from trusted financial websites.
- **Evidence Refinement:** Selects relevant evidence passages for answer generation.
- **Numerical Verification:** Checks numerical claims against supporting evidence.
- **PDF Processing:** Extracts and processes text and tables from financial documents.
- **Chart Understanding:** Uses a vision-language model to interpret chart-based information.
- **Compound Query Handling:** Decomposes complex questions into smaller sub-questions.
- **Evaluation Dashboard:** Displays system evaluation results and performance metrics through Streamlit.

## Problem Statement

Traditional RAG systems may retrieve irrelevant or incomplete information and generate answers without sufficiently checking the quality of the supporting evidence.

In financial applications, inaccurate figures, outdated information, and unsupported claims can reduce answer reliability.

FinSight addresses these challenges by introducing a corrective retrieval workflow that evaluates evidence, uses trusted official sources when required, and verifies numerical information before presenting answers.

## System Architecture

The FinSight pipeline follows a modular retrieval and verification workflow:

```text
             User Query
                  |
                  v
          Query Guard Module
                  |
                  v
         Query Decomposition
                  |
                  v
          Retrieval Module
                  |
          +-------+-------+
          |               |
          v               v
      Dense Search     BM25 Search
          |               |
          +-------+-------+
                  |
                  v
          Relevance Evaluation
                  |
                  v
          Retrieval Decision
                  |
          +-------+-------+
          |               |
          v               v
     Local Evidence    Official Web
       Refinement       Retrieval
          |               |
          +-------+-------+
                  |
                  v
          Answer Generation
                  |
                  v
         Numerical Verification
                  |
                  v
           Final Response
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3 |
| Frontend | Streamlit |
| Visualization | Plotly |
| Vector Database | ChromaDB |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranking Model | ms-marco-MiniLM-L-6-v2 |
| Keyword Retrieval | BM25 (rank_bm25) |
| LLM | GPT-OSS-120B through Groq API |
| Vision Model | Qwen Vision-Language Model |
| PDF Processing | PyMuPDF, pypdf |
| Web Retrieval | DuckDuckGo, Requests, BeautifulSoup |
| Data Processing | NumPy, Pandas |

## Project Structure

```text
FinSight/
│
├── app.py
├── frontend.py
├── pipeline.py
│
├── guard.py
├── decompose.py
├── retriever.py
├── evaluator.py
├── trigger.py
├── refine.py
├── web_search.py
├── generator.py
├── verifier.py
├── calculator.py
│
├── ingest.py
├── ingest_tables.py
├── vision.py
├── charts.py
│
├── config.py
├── domains.py
├── llm.py
│
├── evaluate.py
├── auto_grade_v2.py
├── label_tool.py
├── score.py
│
├── data/
│   ├── Banking_rbi/
│   ├── Company_reports/
│   ├── Economy/
│   ├── Mutual_funds/
│   ├── Retirement_nps/
│   ├── Stock_market/
│   └── Tax/
│
├── eval_dev.json
├── eval_final.json
├── eval_set.json
│
├── results_dev_dense_v4.csv
├── results_dev_hybrid_v4.csv
├── results_final.csv
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Data Sources

FinSight uses a collection of financial PDF documents and selected official websites.

**Document categories:**
- Banking and RBI publications
- Company reports
- Economic reports and indicators
- Mutual fund publications
- Retirement and NPS documents
- Stock market educational documents
- Income tax documents

**Official web sources:**

- [Reserve Bank of India](https://www.rbi.org.in/)
- [Securities and Exchange Board of India](https://www.sebi.gov.in/)
- [Association of Mutual Funds in India](https://www.amfiindia.com/)
- [National Stock Exchange](https://www.nseindia.com/)
- [Income Tax Department](https://www.incometax.gov.in/)
- [Ministry of Statistics and Programme Implementation](https://www.mospi.gov.in/)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Divyanshisen/FinSight.git
cd FinSight
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

Add the API credentials required by the application, using the variable names defined in `config.py` and `.env.example`.

**Important:** Never upload your actual `.env` file or API credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access FinSight.

## Evaluation and Results

FinSight was evaluated against a standard RAG baseline using development and final evaluation datasets.

### Final Evaluation Results

| Metric | Standard RAG | FinSight |
|---|---:|---:|
| Answer Accuracy | 57.1% | **85.7%** |
| Reliability | 88.9% | 85.7% |
| Hallucination Rate | 7.1% | 14.3% |
| Live Questions Answered | 2/6 | **6/6** |
| Unanswerable Questions Refused | 4/4 | 4/4 |

### Development Set Performance

| Configuration | Accuracy | Average Time |
|---|---:|---:|
| Standard RAG | 45% | N/A |
| FinSight – Dense Retrieval | 80% | 11.0 sec |
| FinSight – Hybrid Retrieval | 75% | 16.5 sec |

**Key Observations:**

- FinSight achieved higher answer accuracy than the standard RAG baseline on the final evaluation set.
- Official web retrieval improved the handling of live financial questions.
- Dense retrieval achieved 80% accuracy on the development set.
- Hybrid retrieval did not show an improvement over dense retrieval in this experiment.
- FinSight's higher hallucination rate indicates that further improvements in evidence verification are needed.

**Evaluation Dataset:**
- Development set: 24 questions (20 answerable and 4 unanswerable).
- Final set: 18 questions (14 answerable and 4 unanswerable).

*Results are based on the project's recorded evaluation runs and are specific to the tested datasets.*

## Limitations

- Evaluation was conducted on a relatively small question set.
- Official websites may restrict automated retrieval.
- Some chart-based answers require improved visual evidence auditing.
- Complex list-based answers may omit certain details.
- The system depends on external API availability and configured credentials.
- The system is designed for financial information retrieval, not personalized investment advice.

## Future Scope

- Expand the evaluation dataset with human-verified questions.
- Improve table-aware document retrieval.
- Strengthen numerical and citation verification.
- Improve chart interpretation and visual evidence auditing.
- Add more official financial information sources.
- Explore multilingual financial question answering.
- Deploy the application for broader accessibility.

## Disclaimer

FinSight is an academic and research-oriented financial information retrieval project. It provides information based on retrieved documents and available sources.

It does not provide personalized investment recommendations, financial planning, stock price predictions, or real-time trading advice. Users should verify important financial information through official sources.

## Contributors

**Project:** FinSight – A Self-Correcting Retrieval-Augmented Generation Architecture for Financial Data Verification

**Institution:** Madhav Institute of Technology & Science, Gwalior

**Department:** Centre for Artificial Intelligence

**Academic Session:** July–December 2026

## Repository

GitHub: https://github.com/Divyanshisen/FinSight

---

<p align="center">
  <strong>FinSight – Making Financial Information Retrieval More Reliable.</strong>
</p>
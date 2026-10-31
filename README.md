# CRAG Finance

**Corrective Retrieval-Augmented Generation for Finance Q&A**

CRAG Finance is a finance-focused question-answering system that compares standard Retrieval-Augmented Generation (RAG) with a corrective RAG pipeline.

## Key Features

- Retrieval from a local finance-document collection.
- Cross-encoder relevance scoring and corrective retrieval decisions.
- Verified-source web retrieval for supported finance topics.
- Question decomposition for compound queries.
- Local calculation support for supported numerical questions.
- Answer verification and refusal handling.
- Optional chart and visual extraction from supported documents.

## Architecture

1. **Question input** — receive a finance question.
2. **Retrieval** — retrieve relevant local document chunks.
3. **Evaluation** — score question-document relevance.
4. **Decision** — select the corrective retrieval path.
5. **Knowledge refinement** — retain relevant evidence and retrieve official web sources when needed.
6. **Generation and verification** — generate an answer and check its evidence.
7. **Output** — return the answer, sources and verification status.

## Evaluation

The project includes an evaluation set comparing baseline RAG and CRAG. Results are saved to `results_raw.csv`.

Final accuracy, source reliability, latency and chart performance should be reported only after the corresponding evaluation labels and measurements have been checked.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Configure `GROQ_API_KEY` in a local `.env` file.
4. Ensure the required local document collection is available.
5. Run the application using the entry point configured in the project.

**Security:** Never commit `.env`, API keys, tokens or private credentials.
6. Architecture diagram

flowchart TD
    A[User Question] --> B[Retrieve Local Documents]
    B --> C[Cross-Encoder Evaluation]
    C --> D{CRAG Decision}
    D -->|Relevant| E[Refine Documents]
    D -->|Insufficient or Ambiguous| F[Verified Official Web Retrieval]
    E --> G[Generate Answer]
    F --> G
    G --> H[Verify Evidence]
    H --> I[Answer + Sources + Status]

# TM Forum RAG Assistant

A grounded Retrieval-Augmented Generation (RAG) assistant for exploring publicly available TM Forum API specifications.

The application helps telecom analysts, solution architects, and developers ask natural-language questions across multiple TM Forum domains and receive answers grounded in the source documentation with page- and chunk-level citations.

> **25-question evaluation: 92% retrieval relevance and 96% answer faithfulness.**

---

## What Does It Do?

Instead of manually searching through large TM Forum API specification documents, users can ask questions such as:

- What is a BillingAccount?
- How do I retrieve a CustomerBill by ID?
- What fields are mandatory when creating a Payment?
- How do I cancel a ProductOrder?
- How do I update a customer's contact information?
- Which API manages customer information versus billing-account information?

The assistant retrieves relevant documentation, reranks the evidence, checks whether the evidence is sufficient, and then either:

**generates a grounded answer with citations**

or

**refuses to answer when the documentation does not provide enough evidence.**

---

### Source Data and Repository Samples

This project uses TM Forum API specification documents obtained through authorized access to the TM Forum website.

The original PDF specifications and the full derived chunk corpus are intentionally **not included in this public repository**. Users should obtain the applicable TM Forum specifications directly from TM Forum and follow the corresponding access .

The repository includes only a small schema-oriented sample:

`data/sample/sample_chunks.json`

This sample demonstrates the structure produced by the ingestion pipeline, including:

- chunk ID
- contextualized chunk text format
- source document metadata
- API ID and API name
- API version
- source page
- content type

The full local corpus used by the application contains **902 processed chunks across five TM Forum API specifications**. The sample file is provided only to illustrate the data structure expected by the ingestion and retrieval pipeline; it is not the complete knowledge corpus.

The following directories are intentionally excluded from Git:

- `data/raw/` — locally downloaded source documents
- `data/processed/` — full processed/chunked corpus

Both are excluded through `.gitignore`.

---

## Knowledge Base

The project uses only  TM Forum API specifications obtained through authorized access to the TM Forum website.

No proprietary employer, customer, or internal telecom documentation is included.

| API | Domain | Version | Chunks |
|---|---|---:|---:|
| TMF666 | Account Management 
| TMF622 | Product Ordering Management 
| TMF678 | Customer Bill Management 
| TMF676 | Payment Management 
| TMF629 | Customer Management 

---

## RAG Architecture

The final architecture evolved iteratively as retrieval failures were discovered during testing.

```text
                      User Question
                              │
                    Query Type Detection
                              │
                 ┌────────────┴────────────┐
                 │                         │
           Standard Query           Comparison Query
                 │                         │
                 │                  Query Decomposition
                 │                    ↙          ↘
                 │               Entity A      Entity B
                 │                    ↓          ↓
                 └────────────── Hybrid Retrieval
                              │
                  Dense Search + BM25
                              │
                         RRF Fusion
                              │
                        LLM Reranking
                              │
                 Balanced Evidence Selection
                              │
                  LangGraph Evidence Check
                       ↙              ↘
                  Sufficient       Insufficient
                      │                 │
               Grounded Answer       Refusal
                      │
                  Citations

**Retrieval Pipeline**
1. Dense retrieval — semantic vector search using Pinecone
2. BM25 retrieval — exact keyword and technical-term matching
3. Reciprocal Rank Fusion (RRF) — combines dense and sparse rankings
4. LLM reranking — promotes the candidates that most directly answer the question
5. LangGraph evidence check — determines whether retrieved evidence is sufficient
6. Grounded generation — answers only from retrieved documentation
7. Inline citations — identifies the API, source page, and chunk

**Example citation:**
[TMF622 | Page 154 | Chunk TMF622_0293]
Why Hybrid Retrieval?
The project initially used only dense semantic retrieval.
Testing revealed that semantic search often found the correct API but sometimes ranked JSON examples above exact definitions.
BM25 was added to improve exact technical matches such as:
BillingAccount
CustomerBill
POST /payment
Mandatory Attributes
However, BM25 struggled when closely related resources shared similar terminology — for example, Payment and Refund.
The two approaches were therefore combined using hybrid retrieval + Reciprocal Rank Fusion.
LLM reranking was then added because the correct evidence was often present in the candidate set but was not always ranked first.

**Evidence-Based Refusal**
A key design goal was preventing unsupported answers.
LangGraph performs an evidence-sufficiency check before answer generation.
For example:
Question
What is the SLA for resolving a payment dispute?

**Response**
I could not find enough information in the retrieved TM Forum documentation.

The assistant therefore has an explicit answer-or-refuse path rather than generating an answer from weakly related evidence.
Grounding Rules
The answer-generation prompt includes strict grounding requirements.


**Project Structure**
telco-rag/
│
├── data/
│   ├── raw/
│   │   └── TM Forum API PDFs
│   │
│   └── processed/
│       └── *_chunks.json
│
├── evaluation/
│   ├── eval_results.json
│   └── latency_results.json
│
├── src/
│   ├── ingest.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── bm25_retrieve.py
│   ├── hybrid_retrieve.py
│   ├── hybrid_rerank.py
│   ├── answer.py
│   ├── rag_graph.py
│   ├── evaluate.py
│   ├── latency_test.py
│   └── app.py
│
├── notebooks/
│   └── tm_forum_rag_demo.ipynb
│
├── pyproject.toml
└── README.md

## Running the Application
1. Install dependencies
uv sync
2. Configure environment variables
Create a .env file with the API credentials required by the project.
For example:
NEBIUS_API_KEY=...
PINECONE_API_KEY=...

3. Run Streamlit
uv run streamlit run src/app.py
Building the Knowledge Base
Ingest the source PDFs
uv run python src/ingest.py
Create embeddings and upload vectors
uv run python src/embed.py

Evaluation
Run the evaluation suite:
uv run python src/evaluate.py
Run the latency test:
uv run python src/latency_test.py
How the Architecture Evolved
The project followed an earned-complexity approach:
Dense RAG
   ↓
BM25
   ↓
Hybrid + RRF
   ↓
LLM Reranking
   ↓
LangGraph Evidence Gate
   ↓
Comparison-Aware Retrieval

Each layer was added after evaluation exposed a specific failure mode.

Each layer was added only after testing exposed a specific failure mode.
The biggest lesson from the project was:
RAG quality depends heavily on retrieval, grounding, and evaluation — not just the generation model.


Project Context
This project was built as part of a RAG application exercise focused on:
Corpus → Chunking → Embeddings → Storage → Retrieval → Grounded Generation → Evaluation
The Streamlit interface was developed as a vibe-coded UI on top of the RAG backend.


## Data & Licensing Note

TM Forum documentation remains subject to TM Forum's applicable access and licensing terms. This repository contains the implementation, evaluation artifacts, architecture, and a small schema-oriented sample, but intentionally excludes the original TM Forum PDFs and the full derived document corpus.

This project was created for educational and demonstration purposes.

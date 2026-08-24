# TM Forum RAG Assistant

A grounded **Retrieval-Augmented Generation (RAG)** assistant for exploring TM Forum API specifications.

The application helps telecom analysts, solution architects, and developers ask natural-language questions across multiple TM Forum domains and receive answers grounded in source documentation with **API-, page-, and chunk-level citations**.

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

- **generates a grounded answer with citations**, or
- **refuses to answer when the retrieved documentation does not provide enough evidence.**

---

## Knowledge Base

The project uses five TM Forum API specifications obtained through authorized access to the TM Forum website.

No proprietary employer, customer, or internal telecom documentation is included.

| API | Domain | Version | Chunks |
|---|---|---:|---:|
| TMF666 | Account Management | 5.0.0 | 370 |
| TMF678 | Customer Bill Management | 5.0.0 | 102 |
| TMF676 | Payment Management | 4.0.0 | 66 |
| TMF622 | Product Ordering Management | 5.0.0 | 299 |
| TMF629 | Customer Management | 5.0.0 | 65 |
| **Total** | **5 TM Forum APIs** | | **902** |

### Source Data and Repository Samples

The original PDF specifications and the full derived chunk corpus are intentionally **not included in this public repository**.

Users who want to reproduce the complete knowledge base should obtain the applicable specifications directly from TM Forum and follow the corresponding access and licensing terms.

The repository includes only a small schema-oriented sample:

`data/sample/sample_chunks.json`

The sample demonstrates the structure produced by the ingestion pipeline, including:

- chunk ID
- contextualized chunk text format
- source document metadata
- API ID and API name
- API version
- source page
- content type

The full local corpus used by the application contains **902 processed chunks across five TM Forum API specifications**.

The following directories are intentionally excluded from Git:

- `data/raw/` — locally obtained source documents
- `data/processed/` — full processed/chunked corpus

Both are excluded through `.gitignore`.

---

## RAG Architecture

The architecture evolved iteratively as evaluation exposed specific retrieval and grounding failure modes.

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
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| PDF ingestion | Python document loader |
| Chunking | LangChain recursive text splitting |
| Embeddings | Nebius / Qwen3-Embedding-8B |
| Vector database | Pinecone |
| Sparse retrieval | BM25 |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Nebius-hosted instruction LLM |
| Generation | Nebius-hosted instruction LLM |
| Orchestration | LangGraph |
| Evaluation | Custom Python evaluation |
| UI | Streamlit |

---

## Retrieval Pipeline

1. **Dense retrieval** — semantic vector search using Pinecone.
2. **BM25 retrieval** — exact keyword and technical-term matching.
3. **Reciprocal Rank Fusion (RRF)** — combines dense and sparse rankings.
4. **LLM reranking** — promotes candidates that most directly answer the question.
5. **Comparison-aware retrieval** — decomposes multi-entity comparison questions and preserves evidence for both entities.
6. **LangGraph evidence check** — determines whether the retrieved evidence is sufficient.
7. **Grounded generation** — answers only from retrieved documentation.
8. **Inline citations** — identify the API, source page, and chunk supporting the answer.

Example citation:

```text
[TMF622 | Page 154 | Chunk TMF622_0293]
```

---

## Why Hybrid Retrieval?

The project initially used only dense semantic retrieval.

Testing revealed that semantic search often found the correct API but sometimes ranked JSON examples above exact definitions.

BM25 was added to improve exact technical matches such as:

- `BillingAccount`
- `CustomerBill`
- `POST /payment`
- `Mandatory Attributes`

However, BM25 alone struggled when closely related resources shared similar terminology, such as **Payment** and **Refund**.

The two approaches were therefore combined using **hybrid retrieval + Reciprocal Rank Fusion (RRF)**.

LLM reranking was then added because the correct evidence was often present in the candidate set but was not always ranked first.

---

## Comparison-Aware Retrieval

Evaluation exposed another retrieval failure mode: comparison questions could retrieve strong evidence for one entity while under-representing the other.

Examples included:

- BillingAccount vs Customer
- Payment vs Refund
- CustomerBill vs CustomerBillOnDemand

To address this, comparison-aware retrieval was added.

For comparison questions, the system:

1. detects that the question compares multiple entities;
2. decomposes the question into focused entity-specific searches;
3. retrieves evidence for both sides using the hybrid pipeline;
4. preserves evidence representing both entities;
5. merges and deduplicates the candidates;
6. reranks the evidence against the original comparison question; and
7. passes the balanced evidence to the LangGraph sufficiency check.

This improvement was added directly in response to evaluation findings.

---

## Evidence-Based Refusal

A key design goal was preventing unsupported answers.

LangGraph performs an **evidence-sufficiency check** before answer generation.

For example:

> **Question:** What is the SLA for resolving a payment dispute?

When sufficient supporting evidence is not available, the assistant responds:

> I could not find enough information in the retrieved TM Forum documentation.

The assistant therefore has an explicit **answer-or-refuse path** rather than generating an answer from weakly related evidence.

---

## Grounding Rules

The answer-generation prompt applies strict grounding requirements.

The assistant is instructed to:

- use only retrieved evidence;
- distinguish closely related resources;
- avoid unsupported domain inference;
- distinguish mandatory fields from fields merely shown in examples;
- avoid treating example values as complete enumerations;
- preserve exact API-operation distinctions;
- align factual claims with their citations; and
- provide a limited answer rather than filling evidence gaps with assumptions.

---

## Evaluation

The application was evaluated using a **25-question test set** covering:

- definitions;
- API operations;
- exact and mandatory fields;
- cross-document reasoning;
- multi-entity comparisons;
- unsupported questions; and
- scope-boundary behavior.

### Evaluation-Driven Improvement

Evaluation was used not only to measure the system, but also to drive architectural changes.

One important failure pattern involved multi-entity comparison questions. This led to the implementation of **comparison-aware query decomposition and balanced evidence retrieval**.

The development cycle therefore followed:

```text
Test
  ↓
Identify Failure
  ↓
Improve Retrieval / Grounding
  ↓
Retest
```

A remaining limitation is **scope-boundary reasoning**, where a technically related API operation can sometimes be mistaken for the broader business process being asked about.

Evaluation artifacts are available under:

`evaluation/`

---

## Latency

A five-question latency test measured end-to-end RAG response time.

| Metric | Result |
|---|---:|
| Average latency | 8.45 s |
| Median latency | 7.48 s |
| Minimum latency | 5.68 s |
| Maximum latency | 12.68 s |

The latency reflects the complete pipeline, including retrieval, reranking, evidence validation, and grounded generation.

---

## Project Structure

```text
telco-rag/
│
├── assets/
│   ├── grounded-answer.png
│   └── project-insight.png
│
├── data/
│   └── sample/
│       └── sample_chunks.json
│
├── docs/
│   └── Divya Sahni- Week-2 Project- TM Forum-RAG Assistant.docx
│
├── evaluation/
│   ├── eval_results.json
│   └── latency_results.json
│
├── notebooks/
│   └── tm_forum_rag_demo_with_diagrams.ipynb
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
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

> The complete `data/raw/` and `data/processed/` directories are maintained locally and intentionally excluded from the public repository.

---

## Running the Application

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file containing the required API credentials:

```text
NEBIUS_API_KEY=...
PINECONE_API_KEY=...
```

Never commit the `.env` file to Git.

### 3. Prepare the Knowledge Corpus

The complete TM Forum source documents are not distributed with this repository.

To reproduce the full application:

1. obtain the applicable TM Forum API specifications directly from TM Forum;
2. place the locally obtained PDF files in `data/raw/`;
3. run the ingestion pipeline; and
4. generate embeddings and populate the Pinecone index.

Run ingestion:

```bash
uv run python src/ingest.py
```

Create embeddings:

```bash
uv run python src/embed.py
```

### 4. Run the Streamlit Application

```bash
uv run streamlit run src/app.py
```

---

## Running Evaluation

Run the evaluation suite:

```bash
uv run python src/evaluate.py
```

Run the latency test:

```bash
uv run python src/latency_test.py
```

---

## How the Architecture Evolved

The project followed an **earned-complexity approach**:

```text
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
```

Each layer was added after evaluation exposed a specific failure mode.

The biggest lesson from the project was:

> **RAG quality depends heavily on retrieval, grounding, and evaluation — not just the generation model.**

---

## Project Context

This project was built as part of a RAG application exercise focused on:

**Corpus → Chunking → Embeddings → Storage → Retrieval → Grounded Generation → Evaluation**

The Streamlit interface provides an interactive layer over the RAG backend for exploring the indexed documentation and inspecting grounded answers and supporting evidence.

---

## Data & Licensing Note

TM Forum documentation remains subject to TM Forum's applicable access and licensing terms.

This public repository contains the implementation, architecture, evaluation artifacts, and a small schema-oriented sample, but intentionally excludes:

- the original TM Forum PDF specifications; and
- the complete derived document corpus.

This project was created for **educational and demonstration purposes**.

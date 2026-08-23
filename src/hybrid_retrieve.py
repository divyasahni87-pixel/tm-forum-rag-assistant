import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi


load_dotenv()


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

NEBIUS_MODEL = "Qwen/Qwen3-Embedding-8B"
PINECONE_INDEX_NAME = "telco-rag"

JSON_FILES = [
    "data/processed/TMF666_chunks.json",
    "data/processed/TMF678_chunks.json",
    "data/processed/TMF676_chunks.json",
    "data/processed/TMF622_chunks.json",
    "data/processed/TMF629_chunks.json",
]

DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5

RRF_K = 60


# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index = pc.Index(PINECONE_INDEX_NAME)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():

    all_chunks = []

    for file_path in JSON_FILES:

        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        all_chunks.extend(chunks)

    return all_chunks


chunks = load_chunks()

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in chunks
}


# --------------------------------------------------
# BM25 TOKENIZER
# --------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "how", "do", "does",
    "did", "i", "me", "my", "when", "where", "why",
    "of", "to", "for", "in", "on", "and", "or",
    "with", "be", "by",
}


def tokenize(text):

    raw_tokens = re.findall(
        r"[A-Za-z0-9_@./-]+",
        text
    )

    tokens = []

    for token in raw_tokens:

        normalized = token.lower().strip("./-")

        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)

        # BillingAccount -> billing + account
        # CustomerBill -> customer + bill
        parts = re.findall(
            r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+",
            token
        )

        if len(parts) > 1:

            for part in parts:

                part = part.lower()

                if part not in STOPWORDS:
                    tokens.append(part)

    return tokens


tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------------------------
# DENSE RETRIEVAL
# --------------------------------------------------

def dense_search(question):

    response = nebius_client.embeddings.create(
        model=NEBIUS_MODEL,
        input=question,
    )

    query_vector = response.data[0].embedding

    results = index.query(
        vector=query_vector,
        top_k=DENSE_TOP_K,
        include_metadata=True,
    )

    return [
        match.id
        for match in results.matches
    ]


# --------------------------------------------------
# BM25 RETRIEVAL
# --------------------------------------------------

def bm25_search(question):

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:BM25_TOP_K]

    return [
        chunks[i]["chunk_id"]
        for i in ranked_indexes
    ]


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
):

    scores = {}

    for rank, chunk_id in enumerate(
        dense_results,
        start=1,
    ):

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, chunk_id in enumerate(
        bm25_results,
        start=1,
    ):

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:FINAL_TOP_K]


# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(question):

    dense_results = dense_search(question)
    bm25_results = bm25_search(question)

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )

    return fused_results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def print_results(question, results):

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    for rank, (chunk_id, rrf_score) in enumerate(
        results,
        start=1,
    ):

        chunk = chunk_lookup[chunk_id]
        metadata = chunk["metadata"]

        print(f"\nRESULT {rank}")
        print("-" * 80)

        print(f"RRF Score: {rrf_score:.6f}")
        print(f"Chunk ID: {chunk_id}")
        print(f"API: {metadata.get('api_id')}")
        print(f"API Name: {metadata.get('api_name')}")
        print(f"Version: {metadata.get('version')}")
        print(
            f"Page: "
            f"{metadata.get('page_label', metadata.get('page'))}"
        )
        print(
            f"Content Type: "
            f"{metadata.get('content_type')}"
        )

        print("\nTEXT:")
        print(chunk["text"])


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    questions = [
        "What is a BillingAccount?",
        "How do I retrieve a customer bill?",
        "What fields are mandatory when creating a payment?",
        "What is a ProductOrder?",
        "How do I cancel a ProductOrder?",
        "What actions can be performed on a ProductOrderItem?",
        "What is a Customer?",
        "How do I update a customer's contact information?",
        "Which API manages customer information versus billing-account information?",
    ]

    for question in questions:

        results = hybrid_search(question)

        print_results(question, results)
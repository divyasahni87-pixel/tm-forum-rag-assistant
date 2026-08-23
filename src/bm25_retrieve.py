import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


JSON_FILES = [
    "data/processed/TMF666_chunks.json",
    "data/processed/TMF678_chunks.json",
    "data/processed/TMF676_chunks.json",
    "data/processed/TMF622_chunks.json",
    "data/processed/TMF629_chunks.json",
]

TOP_K = 5

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "who",
    "how",
    "do",
    "does",
    "did",
    "i",
    "me",
    "my",
    "when",
    "where",
    "why",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "with",
    "be",
    "by",
}

# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():
    all_chunks = []

    for file_path in JSON_FILES:
        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {path.name}")
        all_chunks.extend(chunks)

    return all_chunks


# --------------------------------------------------
# TOKENIZE
# --------------------------------------------------

# def tokenize(text):
#     """
#     Basic tokenizer for BM25.
#     Keeps technical terms searchable while normalizing case.
#     """

#     text = text.lower()

#     return re.findall(
#         r"[a-z0-9_@./?-]+",
#         text
#     )


def tokenize(text):
    """
    Tokenizer optimized for technical/API documentation.

    - strips punctuation
    - removes common stopwords
    - preserves technical terms
    - expands camelCase / PascalCase terms
      e.g. CustomerBill -> customerbill + customer + bill
    """

    # Find words before lowercasing so CamelCase is visible
    raw_tokens = re.findall(
        r"[A-Za-z0-9_@./-]+",
        text
    )

    tokens = []

    for token in raw_tokens:

        # Preserve normalized complete token
        normalized = token.lower().strip("./-")

        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)

        # Split CamelCase / PascalCase:
        # BillingAccount -> Billing + Account
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


# --------------------------------------------------
# BUILD BM25 INDEX
# --------------------------------------------------

chunks = load_chunks()

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

print(f"\nBM25 index created for {len(chunks)} chunks")


# --------------------------------------------------
# SEARCH
# --------------------------------------------------

def search_bm25(question, top_k=TOP_K):

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indexes:
        results.append(
            {
                "score": scores[index],
                "chunk": chunks[index],
            }
        )

    return results


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

def print_results(question, results):

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]
        metadata = chunk["metadata"]

        print(f"\nRESULT {rank}")
        print("-" * 80)

        print(f"BM25 Score: {result['score']:.4f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
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
# TEST QUESTIONS
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

        results = search_bm25(question)

        print_results(question, results)
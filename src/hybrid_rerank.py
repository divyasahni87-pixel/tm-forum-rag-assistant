import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi


# Load `NEBIUS_API_KEY` and `PINECONE_API_KEY` from `.env` before clients are
# created. The credentials remain outside the source code.
load_dotenv()


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# The embedding model turns the user's question into a vector for dense search.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
# The chat model reviews the combined candidate set and chooses final evidence.
RERANK_MODEL = "Qwen/Qwen3-32B"

PINECONE_INDEX_NAME = "telco-rag"

# The local corpus is used to build the BM25 keyword-search index and to look
# up full text after Pinecone returns vector IDs.
JSON_FILES = [
    "data/processed/TMF666_chunks.json",
    "data/processed/TMF678_chunks.json",
    "data/processed/TMF676_chunks.json",
    "data/processed/TMF622_chunks.json",
    "data/processed/TMF629_chunks.json",
]

# Retrieve independent candidate lists from semantic (dense) and keyword
# (BM25) search before combining them.
DENSE_TOP_K = 10
BM25_TOP_K = 10

# RRF gives a candidate pool.
RRF_CANDIDATES = 12

# LLM decides final evidence.
FINAL_TOP_K = 5

# Comparison queries retain candidates from each entity before reranking so
# evidence for one resource cannot crowd out the other.
COMPARISON_CANDIDATES_PER_ENTITY = 10

# RRF's rank constant reduces the difference between nearby result positions.
RRF_K = 60

# Enable only when inspecting comparison-query internals during development.
DEBUG = False


# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

# Nebius provides an OpenAI-compatible API for embeddings and chat completion.
nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# Pinecone holds the dense vectors created by `embed.py`.
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index = pc.Index(PINECONE_INDEX_NAME)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():
    """Load all per-API chunk files into the local corpus for BM25 and lookup."""

    all_chunks = []

    for file_path in JSON_FILES:

        path = Path(file_path)

        # Each record has `chunk_id`, contextualized `text`, and `metadata`.
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        all_chunks.extend(chunks)

    return all_chunks


# The JSON corpus is read once when this script starts.
chunks = load_chunks()

# Pinecone returns IDs, so this dictionary restores the original chunk and its
# metadata in constant time during reranking and display.
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
    """Tokenize text for BM25 while preserving API paths and identifiers."""

    raw_tokens = re.findall(
        r"[A-Za-z0-9_@./-]+",
        text
    )

    tokens = []

    for token in raw_tokens:

        # Use normalized full tokens for exact matches such as `POST`, IDs, and
        # endpoint paths, after discarding common words that add little signal.
        normalized = token.lower().strip("./-")

        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)

        # Expand CamelCase:
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


# BM25 works with token lists rather than raw text, so prepare the corpus once.
tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)


# --------------------------------------------------
# DENSE SEARCH
# --------------------------------------------------

def dense_search(question):
    """Return the IDs of chunks nearest to the question in vector space."""

    response = nebius_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    # A single question yields one embedding vector.
    query_vector = response.data[0].embedding

    results = index.query(
        vector=query_vector,
        top_k=DENSE_TOP_K,
        include_metadata=True,
    )

    # Retrieval returns Pinecone IDs only; their full records live locally.
    return [
        match.id
        for match in results.matches
    ]


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(question):
    """Return IDs of chunks with the strongest lexical keyword matches."""

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    # BM25 returns a score per local corpus item; sort its indexes by score.
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
# RRF
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
):
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion (RRF)."""

    scores = {}

    # A chunk in both lists receives contributions from both rankings.
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

    # Retain a slightly broader candidate set for the LLM to inspect.
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        chunk_id
        for chunk_id, _ in ranked[:RRF_CANDIDATES]
    ]


# --------------------------------------------------
# LLM RERANK
# --------------------------------------------------

def llm_rerank(question, candidate_ids, comparison_entities=None):
    """Ask the LLM to rank fused candidates by direct relevance to the question."""

    candidate_text = ""
    comparison_guidance = ""

    if comparison_entities:
        comparison_guidance = """
For comparison questions, the final ranked evidence must collectively
cover both entities being compared.

Prefer direct concept/definition chunks for each entity when they are
available.

A good comparison candidate set should normally contain at least one
strong explanatory chunk for the left entity and one strong explanatory
chunk for the right entity.

Prefer concept or field-definition evidence that explains what the
resource is over JSON examples, CRUD operations, or event examples when
the question asks for the difference between two concepts.

Do not allow several examples about one entity to displace the strongest
definition evidence for the other entity.
"""

    # Give the reranker the source text plus enough metadata to distinguish
    # similarly named resources across the TM Forum API documents.
    for chunk_id in candidate_ids:

        chunk = chunk_lookup[chunk_id]
        metadata = chunk["metadata"]

        candidate_text += f"""
CHUNK_ID: {chunk_id}
API: {metadata.get("api_id")}
CONTENT_TYPE: {metadata.get("content_type")}
TEXT:
{chunk["text"]}

---
"""

    prompt = f"""
You are a relevance reranker for technical TM Forum API documentation.

USER QUESTION:
{question}

CANDIDATE CHUNKS:
{candidate_text}

Rank the chunks by how directly they provide evidence needed
to answer the user's exact question.

Important ranking rules:

1. Prefer the exact API resource named or implied by the question.
2. Prefer the exact operation when the question asks how to create,
   retrieve, update, delete, or list something.
3. Prefer chunks containing the exact requested fields or rules.
4. Distinguish closely related resources carefully.
   For example:
   - Payment is not Refund.
   - CustomerBill is not CustomerBillOnDemand.
5. Prefer direct definitions for "What is..." questions.
6. JSON examples are useful when the question asks about payloads,
   but explanatory definitions or operation rules may be better
   for conceptual questions.
7. Do not answer the user's question.

{comparison_guidance}

8. Return ONLY valid JSON with the exact `ranked_chunk_ids` key in this format:

{{
  "ranked_chunk_ids": [
    "CHUNK_ID_1",
    "CHUNK_ID_2",
    "CHUNK_ID_3",
    "CHUNK_ID_4",
    "CHUNK_ID_5"
  ]
}}
"""

    # JSON mode makes the model response machine-readable for the next step.
    response = nebius_client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    # Debugging guard
    if not content:
        raise ValueError(
            "Reranker returned empty content. "
            f"Full response: {response}"
        )

    content = content.strip()

    # print("\nRAW RERANKER OUTPUT:")
    # print(content)

    # Convert the JSON response into the ordered list requested in the prompt.
    result = json.loads(content)

    ranked_ids = result["ranked_chunk_ids"]

    # Safety check:
    # only keep IDs that were actually candidates.
    valid_ids = [
        chunk_id
        for chunk_id in ranked_ids
        if chunk_id in candidate_ids
    ]

    # Do not allow the model to introduce unknown IDs; cap the final evidence.
    return valid_ids[:FINAL_TOP_K]


# --------------------------------------------------
# COMPLETE RETRIEVAL PIPELINE
# --------------------------------------------------

def _clean_comparison_entity(entity):
    """Remove conversational wrappers without altering technical identifiers."""

    entity = entity.strip().rstrip("?.!,;:")
    entity = re.sub(r"^(?:a|an|the)\s+", "", entity, flags=re.IGNORECASE)

    return entity.strip()


def detect_comparison(question):
    """Return two compared entities for a small set of common query forms."""

    question = question.strip()

    patterns = [
        r"(?:what\s+is\s+)?the\s+difference\s+between\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)\s*[?.!]*$",
        r"compare\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+?)\s*[?.!]*$",
        r"how\s+(?:is|does)\s+(.+?)\s+(?:different\s+from|differ\s+from)\s+(.+?)\s*[?.!]*$",
        r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*[?.!]*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)

        if match:
            left = _clean_comparison_entity(match.group(1))
            right = _clean_comparison_entity(match.group(2))

            if left and right:
                return left, right

    return None


def _comparison_candidates(entity):
    """Run the existing hybrid retrieval steps for one focused entity query."""

    focused_question = f"What is {entity}?"
    dense_results = dense_search(focused_question)
    bm25_results = bm25_search(focused_question)

    return reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )[:COMPARISON_CANDIDATES_PER_ENTITY]


def _merge_unique_candidates(*candidate_lists):
    """Keep the first, highest-ranked occurrence of each chunk ID."""

    merged = []
    seen = set()

    for candidate_ids in candidate_lists:
        for chunk_id in candidate_ids:
            if chunk_id not in seen:
                merged.append(chunk_id)
                seen.add(chunk_id)

    return merged


def _comparison_anchor(entity_name, candidate_ids, excluded_ids=()):
    """Choose the best entity-relevant descriptive focused candidate."""

    normalized_entity = entity_name.casefold()
    relevant_candidates = [
        chunk_id
        for chunk_id in candidate_ids
        if chunk_id not in excluded_ids
        and normalized_entity in chunk_lookup[chunk_id]["text"].casefold()
    ]

    # Candidate order already reflects the focused hybrid retrieval ranking.
    # Search in descriptive-content priority while preserving that order inside
    # each content type.
    for content_type in (
        "concept",
        "field_definition",
        "operation",
        "json_example",
    ):
        for chunk_id in relevant_candidates:
            candidate_type = (
                chunk_lookup[chunk_id]["metadata"].get("content_type") or ""
            ).lower()

            if candidate_type == content_type:
                return chunk_id

    # If no focused candidate explicitly names this entity, retain the original
    # highest-ranked focused result rather than selecting an unrelated chunk.
    for chunk_id in candidate_ids:
        if chunk_id not in excluded_ids:
            return chunk_id

    return None


def _comparison_final_results(left_anchor, right_anchor, reranked_ids):
    """Keep one anchor for each entity, then fill the remaining final slots."""

    anchor_ids = [
        chunk_id
        for chunk_id in (left_anchor, right_anchor)
        if chunk_id is not None
    ]

    return _merge_unique_candidates(
        anchor_ids,
        reranked_ids,
    )[:FINAL_TOP_K]


def hybrid_rerank_search(question):
    """Run dense search, BM25 search, RRF fusion, then LLM reranking."""

    comparison = detect_comparison(question)

    if comparison:
        left_entity, right_entity = comparison

        if DEBUG:
            print("COMPARISON QUERY DETECTED")
            print(f"Left entity: {left_entity}")
            print(f"Right entity: {right_entity}")

        # Retrieve each side separately, retain a balanced candidate set, and
        # rerank it against the original comparison question.
        left_candidates = _comparison_candidates(left_entity)
        right_candidates = _comparison_candidates(right_entity)
        left_anchor = _comparison_anchor(left_entity, left_candidates)
        right_anchor = _comparison_anchor(
            right_entity,
            right_candidates,
            excluded_ids=(left_anchor,),
        )

        if DEBUG:
            left_content_type = (
                chunk_lookup[left_anchor]["metadata"].get("content_type")
                if left_anchor else "none"
            )
            right_content_type = (
                chunk_lookup[right_anchor]["metadata"].get("content_type")
                if right_anchor else "none"
            )
            print(f"Selected left anchor: {left_anchor} ({left_content_type})")
            print(f"Selected right anchor: {right_anchor} ({right_content_type})")
        comparison_candidates = _merge_unique_candidates(
            left_candidates,
            right_candidates,
        )

        reranked_results = llm_rerank(
            question,
            comparison_candidates,
            comparison_entities=comparison,
        )

        return _comparison_final_results(
            left_anchor,
            right_anchor,
            reranked_results,
        )

    # Dense and BM25 retrieval provide complementary semantic and exact-term
    # signals before RRF merges their ranked ID lists.
    dense_results = dense_search(question)

    bm25_results = bm25_search(question)

    rrf_candidates = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
    )

    reranked_results = llm_rerank(
        question,
        rrf_candidates,
    )

    return reranked_results


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

def print_results(question, result_ids):
    """Print the final ranked chunks with their source metadata for inspection."""

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    for rank, chunk_id in enumerate(
        result_ids,
        start=1,
    ):

        chunk = chunk_lookup[chunk_id]
        metadata = chunk["metadata"]

        print(f"\nRESULT {rank}")
        print("-" * 80)

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

    # Example questions make this file directly runnable for retrieval testing.
    questions = [
        "What is the difference between a BillingAccount and a Customer?",
        "What is the difference between a Payment and a Refund?",
        "What is the difference between CustomerBill and CustomerBillOnDemand?",
        "What is a BillingAccount?",
    ]

    for question in questions:

        results = hybrid_rerank_search(question)

        print_results(question, results)

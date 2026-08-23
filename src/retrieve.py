import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


load_dotenv()

NEBIUS_MODEL = "Qwen/Qwen3-Embedding-8B"
PINECONE_INDEX_NAME = "telco-rag"

TOP_K = 5


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
# EMBED USER QUERY
# --------------------------------------------------

def embed_query(question):

    response = nebius_client.embeddings.create(
        model=NEBIUS_MODEL,
        input=question,
    )

    return response.data[0].embedding


# --------------------------------------------------
# RETRIEVE FROM PINECONE
# --------------------------------------------------

def retrieve(question, top_k=TOP_K):

    query_vector = embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def print_results(question, results):

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    for rank, match in enumerate(results.matches, start=1):

        metadata = match.metadata or {}

        print(f"\nRESULT {rank}")
        print("-" * 80)

        print(f"Score: {match.score:.4f}")
        print(f"Chunk ID: {match.id}")
        print(f"API: {metadata.get('api_id')}")
        print(f"API Name: {metadata.get('api_name')}")
        print(f"Version: {metadata.get('version')}")
        print(f"Page: {metadata.get('page_label', metadata.get('page'))}")
        print(f"Content Type: {metadata.get('content_type')}")

        print("\nTEXT:")
        print(metadata.get("text", ""))


# --------------------------------------------------
# TEST QUESTIONS
# --------------------------------------------------

if __name__ == "__main__":

    questions = [
        "What is a BillingAccount?",
        "How do I retrieve a customer bill?",
        "What fields are mandatory when creating a payment?",
    ]

    for question in questions:

        results = retrieve(question)

        print_results(question, results)
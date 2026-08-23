import os

from dotenv import load_dotenv
from openai import OpenAI

from hybrid_rerank import hybrid_rerank_search, chunk_lookup


load_dotenv()

ANSWER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)


def build_context(chunk_ids):
    context_parts = []

    for chunk_id in chunk_ids:
        chunk = chunk_lookup[chunk_id]
        metadata = chunk["metadata"]

        page = (
            metadata.get("page_label")
            or metadata.get("page")
        )

        context_parts.append(
            f"""
SOURCE
Chunk ID: {chunk_id}
API: {metadata.get("api_id")}
API Name: {metadata.get("api_name")}
Version: {metadata.get("version")}
Page: {page}

CONTENT:
{chunk["text"]}
"""
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question):

    chunk_ids = hybrid_rerank_search(question)

    context = build_context(chunk_ids)

    prompt = f"""
You are a grounded assistant for TM Forum API documentation.

Answer the user's question using ONLY the supplied source chunks.

USER QUESTION:
{question}

SOURCE CHUNKS:
{context}

RULES:

1. Use only information supported by the source chunks.
2. Do not invent fields, operations, behavior, or API details.
3. Prefer evidence about the exact resource in the question.
   For example:
   - CustomerBill is not CustomerBillOnDemand.
   - Payment is not Refund.
4. If the supplied chunks do not contain enough information to answer,
   say:
   "I could not find enough information in the retrieved TM Forum documentation."
5. Keep the answer concise but useful.
6. Cite claims inline using this format:
   [TMFxxx | Page X | Chunk TMFxxx_XXXX]
7. If multiple chunks support the answer, cite the most direct source.
"""

    response = nebius_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()

    return answer, chunk_ids


if __name__ == "__main__":

    questions = [
        "What is a BillingAccount?",
        "How do I retrieve a customer bill?",
        "What fields are mandatory when creating a payment?",
        "What is the SLA for resolving a payment dispute?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        answer, source_ids = generate_answer(question)

        print("\nANSWER:")
        print(answer)

        print("\nRETRIEVED SOURCES:")
        print(", ".join(source_ids))
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# Load credentials from `.env` into environment variables before creating API
# clients. The keys themselves are not stored in this source file.
load_dotenv()

# Nebius exposes an OpenAI-compatible embeddings endpoint, so the OpenAI client
# can be used by supplying Nebius's base URL and embedding model name.
NEBIUS_MODEL = "Qwen/Qwen3-Embedding-8B"
PINECONE_INDEX_NAME = "telco-rag"

# One processed JSON file is produced for each source TM Forum API. All records
# from these files are embedded and uploaded into the same Pinecone index.
JSON_FILES = [
    "data/processed/TMF622_chunks.json",
    "data/processed/TMF629_chunks.json",
]

# Requests sent to the embedding model and vector upserts use independent batch
# sizes. The latter can be larger because it does not call the embedding model.
EMBEDDING_BATCH_SIZE = 20
PINECONE_BATCH_SIZE = 50


# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

# This client sends embedding requests to Nebius, not to OpenAI's default API.
nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# The Pinecone client connects using the index name configured above. The index
# is assumed to have already been created with a matching vector dimension.
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index = pc.Index(PINECONE_INDEX_NAME)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

def load_chunks():
    # Combine the per-API ingestion outputs into one in-memory list for this
    # embedding run. The original JSON files remain separate on disk.
    all_chunks = []

    for file_path in JSON_FILES:
        path = Path(file_path)

        # Each file contains dictionaries with `chunk_id`, `text`, and
        # `metadata`, as written by `ingest.py`.
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {path.name}")
        all_chunks.extend(chunks)

    return all_chunks


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(texts):
    # The API returns one embedding object per input text, in the same order as
    # the supplied `texts` list.
    response = nebius_client.embeddings.create(
        model=NEBIUS_MODEL,
        input=texts,
    )

    return [
        item.embedding
        for item in response.data
    ]


# --------------------------------------------------
# UPLOAD CHUNKS
# --------------------------------------------------

def upload_chunks(chunks):

    # Tracks progress across embedding batches, rather than across Pinecone
    # upsert sub-batches.
    uploaded_count = 0

    for start in range(
        0,
        len(chunks),
        EMBEDDING_BATCH_SIZE,
    ):

        # Send a limited number of chunk texts in one embedding API request.
        batch = chunks[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        # `zip` below pairs each returned vector with its originating chunk.
        embeddings = create_embeddings(texts)

        vectors = []

        for chunk, embedding in zip(
            batch,
            embeddings,
        ):

            # Copy metadata so adding the retrieval text does not mutate the
            # in-memory chunk loaded from the JSON source file.
            metadata = chunk["metadata"].copy()

            # Store chunk text in Pinecone metadata
            # so retrieval can return the actual source text.
            metadata["text"] = chunk["text"]

            # Pinecone requires an ID, numeric vector values, and optional
            # metadata. The chunk ID provides a stable vector ID per API.
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # Upload the current embedding batch
        # to Pinecone in smaller groups.
        for vector_start in range(
            0,
            len(vectors),
            PINECONE_BATCH_SIZE,
        ):

            vector_batch = vectors[
                vector_start:
                vector_start + PINECONE_BATCH_SIZE
            ]

            # Upsert creates missing vector IDs and replaces vectors with the
            # same IDs on a later run, keeping uploads repeatable.
            index.upsert(
                vectors=vector_batch
            )

        uploaded_count += len(batch)

        print(
            f"Embedded and uploaded "
            f"{uploaded_count}/{len(chunks)} chunks"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    # This guard ensures the upload runs only when this file is executed as a
    # script, not when its functions are imported elsewhere.
    chunks = load_chunks()

    print(f"\nTotal chunks to embed: {len(chunks)}")

    upload_chunks(chunks)

    print("\nEmbedding and Pinecone upload complete.")

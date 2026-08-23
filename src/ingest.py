import json
import os
import re

#from langchain_community.document_loaders import PyPDFLoader
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Each configuration supplies the API metadata, source PDF, and the number of
# front-matter pages to skip. Each API is saved to its own JSON output file.
DOCUMENTS = [
    {
        "path": "data/raw/TMF666_Account_management.pdf",
        "api_id": "TMF666",
        "api_name": "Account Management",
        "version": "5.0.0",
        "skip_pages": 5,
    },
    {
        "path": "data/raw/TMF678_Customer_Bill_userguide.pdf",
        "api_id": "TMF678",
        "api_name": "Customer Bill Management",
        "version": "5.0.0",
        "skip_pages": 3,
    },
    {
    "path": "data/raw/TMF676_Payment_Management_API_v4.0.0_specification.pdf",
    "api_id": "TMF676",
    "api_name": "Payment Management",
    "version": "4.0.0",
    "skip_pages": 6
    },
    {
        "path": "data/raw/TMF622_Product_Ordering_userguide.pdf",
        "api_id": "TMF622",
        "api_name": "Product Ordering Management",
        "version": "5.0.0",
        "skip_pages": 3,
    },
    {
        "path": "data/raw/TMF629_Customer_userguide.pdf",
        "api_id": "TMF629",
        "api_name": "Customer Management",
        "version": "5.0.0",
        "skip_pages": 3,
    },
]


def clean_text(text):
    """Remove repeated PDF artifacts before the text is chunked."""

    # These strings appear repeatedly in the source document's header/footer.
    # Removing them prevents them from adding noise to search and embeddings.
    
    # Remove TM Forum copyright/footer lines regardless of year
    text = re.sub(
        r"©\s*TM Forum\s*\d{4}\.?\s*All Rights Reserved\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone page labels
    text = re.sub(
        r"^\s*Page\s+[ivxlcdm\d]+\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Preserve ordinary paragraph breaks, but collapse extra blank lines caused
    # by PDF text extraction.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def classify_chunk(text):
    """
    Classify each chunk into one simple category:
    concept, field_definition, operation, or json_example
    """

    lower_text = text.lower()

    # This is a lightweight, rule-based classifier. Lowercasing lets keyword
    # checks work regardless of the original capitalization.
    json_signals = [
        '"@type"',
        '"href"',
        '"id"',
        '"name"',
        "content-type: application/json",
        "request body",
        "response body",
    ]

    json_score = sum(signal in lower_text for signal in json_signals)

    # JSON may lack the keywords above, so braces and quotes are a second hint.
    brace_count = text.count("{") + text.count("}")
    quote_count = text.count('"')

    if json_score >= 2 or brace_count >= 3 or quote_count >= 12:
        return "json_example"

    # These phrases commonly occur in the resource-property tables.
    field_signals = [
        "field descriptions",
        "fields",
        "a string.",
        "a boolean.",
        "an integer.",
        "a datetime.",
        "a money.",
        "a timeperiod.",
    ]

    if any(signal in lower_text for signal in field_signals):
        return "field_definition"

    # Detect endpoint/operation documentation. This follows JSON detection so
    # an example containing POST or PATCH remains a `json_example`.
    operation_signals = [
        "retrieves a",
        "list or find",
        "creates a",
        "updates partially",
        "deletes a",
        "api operations",
        "operation on",
        "operations on",
        "get ",
        "post ",
        "patch ",
        "delete ",
    ]

    if any(signal in lower_text for signal in operation_signals):
        return "operation"

    return "concept"


def ingest_document(config):
    """Ingest one configured TM Forum PDF and save its chunks separately."""

    api_id = config["api_id"]
    api_name = config["api_name"]

    print("\n" + "=" * 80)
    print(f"Processing {api_id} - {api_name}")
    print("=" * 80)

    # The loader returns one LangChain Document per PDF page, preserving the
    # loader-provided source and page metadata on each Document.
    #loader = PyPDFLoader(config["path"])
    #documents = loader.load()

    loader = OpenDataLoaderPDFLoader(
    file_path=config["path"],
    format="text",
    quiet=True
)

    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    # Remove only the configured front-matter pages for this source PDF.
    documents = documents[config["skip_pages"]:]

    # Clean page text and add API metadata that all derived chunks inherit.
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["api_id"] = api_id
        doc.metadata["api_name"] = api_name
        doc.metadata["version"] = config["version"]

    # Ignore pages that contain too little meaningful content after cleaning.
    documents = [
        doc
        for doc in documents
        if len(doc.page_content.strip()) > 100
    ]
    print(f"Useful pages after cleaning: {len(documents)}")

    # Preserve the existing splitting settings and page-by-page behavior.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)

    # Drop short tail chunks that are unlikely to be useful standalone evidence.
    chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) >= 200
    ]

    # Classify the original text before adding the artificial context prefix.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{api_id}_{i + 1:04}"
        chunk.metadata["content_type"] = classify_chunk(chunk.page_content)

    # Prefix improves retrieval by supplying API, content-type, and page context.
    for chunk in chunks:
        page_reference = (
            chunk.metadata.get("page_label")
            or chunk.metadata.get("page")
            )
        context_prefix = (
            f"{api_id} {api_name} | "
            f"{chunk.metadata['content_type']} | "
            f"Page {page_reference}"
        )
        chunk.page_content = context_prefix + "\n\n" + chunk.page_content

    print(f"Total chunks created: {len(chunks)}")

    # Keep every category, including JSON examples, in this baseline corpus.
    rag_chunks = chunks
    print(f"RAG chunks to embed: {len(rag_chunks)}")
    print("JSON/example chunks are included in the baseline.")

    category_counts = {
        "concept": 0,
        "field_definition": 0,
        "operation": 0,
        "json_example": 0,
    }
    for chunk in chunks:
        category = chunk.metadata["content_type"]
        category_counts[category] += 1

    print("\nChunk categories:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    # Convert LangChain Documents into the existing JSON-friendly structure.
    chunks_for_json = [
        {
            "chunk_id": chunk.metadata["chunk_id"],
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in rag_chunks
    ]

    os.makedirs("data/processed", exist_ok=True)
    output_path = f"data/processed/{api_id}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks_for_json, file, indent=2, ensure_ascii=False)

    print(f"\nSaved chunks to: {output_path}")
    return rag_chunks, category_counts


total_chunks = 0

for document_config in DOCUMENTS:
    processed_chunks, _ = ingest_document(document_config)
    total_chunks += len(processed_chunks)

print("\n" + "=" * 80)
print("INGESTION COMPLETE")
print("=" * 80)
print(f"Total chunks across all APIs: {total_chunks}")

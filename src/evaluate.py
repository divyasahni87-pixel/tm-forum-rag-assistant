import json
import os
from pathlib import Path

from rag_graph import rag_graph
from hybrid_rerank import chunk_lookup


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "eval_results.json"


# --------------------------------------------------
# EVALUATION DATASET
# --------------------------------------------------

EVAL_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What is a BillingAccount?",
        "expected_api": ["TMF666"],
        "should_answer": True,
        "category": "definition",
    },
    {
        "id": "Q02",
        "question": "What is a SettlementAccount?",
        "expected_api": ["TMF666"],
        "should_answer": True,
        "category": "definition",
    },
    {
        "id": "Q03",
        "question": "What is the difference between a BillingAccount and a SettlementAccount?",
        "expected_api": ["TMF666"],
        "should_answer": True,
        "category": "multi_chunk",
    },
    {
        "id": "Q04",
        "question": "How do I retrieve a BillingAccount by ID?",
        "expected_api": ["TMF666"],
        "should_answer": True,
        "category": "operation",
    },
    {
        "id": "Q05",
        "question": "What fields are mandatory when creating a BillingAccount?",
        "expected_api": ["TMF666"],
        "should_answer": True,
        "category": "exact_fields",
    },
    {
        "id": "Q06",
        "question": "What is a CustomerBill?",
        "expected_api": ["TMF678"],
        "should_answer": True,
        "category": "definition",
    },
    {
        "id": "Q07",
        "question": "How do I retrieve a CustomerBill by ID?",
        "expected_api": ["TMF678"],
        "should_answer": True,
        "category": "operation",
    },
    {
        "id": "Q08",
        "question": "What is CustomerBillOnDemand?",
        "expected_api": ["TMF678"],
        "should_answer": True,
        "category": "definition",
    },
    {
        "id": "Q09",
        "question": "What is the difference between CustomerBill and CustomerBillOnDemand?",
        "expected_api": ["TMF678"],
        "should_answer": True,
        "category": "multi_chunk",
    },
    {
        "id": "Q10",
        "question": "What fields are mandatory when creating a Payment?",
        "expected_api": ["TMF676"],
        "should_answer": True,
        "category": "exact_fields",
    },
    {
        "id": "Q11",
        "question": "How do I retrieve a Payment by ID?",
        "expected_api": ["TMF676"],
        "should_answer": True,
        "category": "operation",
    },
    {
        "id": "Q12",
        "question": "What is the difference between a Payment and a Refund?",
        "expected_api": ["TMF676"],
        "should_answer": True,
        "category": "multi_chunk",
    },
    {
        "id": "Q13",
        "question": "How are Payment and CustomerBill related?",
        "expected_api": ["TMF676", "TMF678"],
        "should_answer": True,
        "category": "cross_document",
    },
    {
        "id": "Q14",
        "question": "What is the SLA for resolving a payment dispute?",
        "expected_api": [],
        "should_answer": False,
        "category": "unsupported",
    },
    {
        "id": "Q15",
        "question": "What API should I use to cancel a mobile phone subscription?",
        "expected_api": [],
        "should_answer": False,
        "category": "unsupported",
    },
    {
    "id": "Q16",
    "question": "What is a ProductOrder?",
    "expected_api": ["TMF622"],
    "should_answer": True,
    "category": "definition",
},
{
    "id": "Q17",
    "question": "How do I retrieve a ProductOrder by ID?",
    "expected_api": ["TMF622"],
    "should_answer": True,
    "category": "operation",
},
{
    "id": "Q18",
    "question": "How do I cancel a ProductOrder?",
    "expected_api": ["TMF622"],
    "should_answer": True,
    "category": "operation_distinction",
},
{
    "id": "Q19",
    "question": "What fields are mandatory when creating a CancelProductOrder?",
    "expected_api": ["TMF622"],
    "should_answer": True,
    "category": "exact_fields",
},
{
    "id": "Q20",
    "question": "What actions can be performed on a ProductOrderItem?",
    "expected_api": ["TMF622"],
    "should_answer": True,
    "category": "enumeration_grounding",
},
{
    "id": "Q21",
    "question": "What is a Customer?",
    "expected_api": ["TMF629"],
    "should_answer": True,
    "category": "definition",
},
{
    "id": "Q22",
    "question": "How do I retrieve a Customer by ID?",
    "expected_api": ["TMF629"],
    "should_answer": True,
    "category": "operation",
},
{
    "id": "Q23",
    "question": "What fields are mandatory when creating a Customer?",
    "expected_api": ["TMF629"],
    "should_answer": True,
    "category": "exact_fields",
},
{
    "id": "Q24",
    "question": "How do I update a customer's contact information?",
    "expected_api": ["TMF629"],
    "should_answer": True,
    "category": "update",
},
{
    "id": "Q25",
    "question": "Which API manages customer information versus billing-account information?",
    "expected_api": ["TMF629", "TMF666"],
    "should_answer": True,
    "category": "cross_document",
},
]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_retrieved_apis(chunk_ids):
    """
    Convert retrieved chunk IDs into the API IDs
    represented in those chunks.
    """

    api_ids = []

    for chunk_id in chunk_ids:
        chunk = chunk_lookup.get(chunk_id)

        if not chunk:
            continue

        api_id = chunk["metadata"].get("api_id")

        if api_id and api_id not in api_ids:
            api_ids.append(api_id)

    return api_ids


def expected_api_found(expected_apis, retrieved_apis):
    """
    Simple automatic retrieval sanity check.

    For answerable questions:
    At least one expected API should be present.

    For unsupported questions:
    This metric is not used for correctness because
    retrieval may still return semantically similar chunks.
    """

    if not expected_apis:
        return None

    return all(
        api in retrieved_apis
        for api in expected_apis
    )


# --------------------------------------------------
# RUN ONE EVALUATION
# --------------------------------------------------

def evaluate_question(item):

    question = item["question"]

    result = rag_graph.invoke(
        {
            "question": question,
            "chunk_ids": [],
            "context": "",
            "evidence_sufficient": False,
            "answer": "",
        }
    )

    chunk_ids = result.get("chunk_ids", [])
    retrieved_apis = get_retrieved_apis(chunk_ids)

    behavior_correct = (
        result["evidence_sufficient"] == item["should_answer"]
    )

    api_check = expected_api_found(
        item["expected_api"],
        retrieved_apis,
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": question,

        "expected_api": item["expected_api"],
        "should_answer": item["should_answer"],

        "retrieved_chunk_ids": chunk_ids,
        "retrieved_apis": retrieved_apis,

        "expected_api_found": api_check,
        "evidence_sufficient": result["evidence_sufficient"],
        "behavior_correct": behavior_correct,

        "answer": result["answer"],

        # Manual review fields.
        # We will fill these in after inspecting results.
        "manual_retrieval_relevant": None,
        "manual_answer_faithful": None,
        "manual_answer_complete": None,
        "manual_citation_correct": None,
        "manual_failure_notes": "",
        
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    for item in EVAL_QUESTIONS:

        print(
            f"\nRunning {item['id']}: "
            f"{item['question']}"
        )

        result = evaluate_question(item)

        results.append(result)

        print(
            f"Evidence sufficient: "
            f"{result['evidence_sufficient']}"
        )

        print(
            f"Retrieved APIs: "
            f"{result['retrieved_apis']}"
        )

        print(
            f"Behavior correct: "
            f"{result['behavior_correct']}"
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\n" + "-" * 80)

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # AUTOMATIC SUMMARY
    # --------------------------------------------------

    total = len(results)

    behavior_correct_count = sum(
        1
        for result in results
        if result["behavior_correct"]
    )

    answerable_results = [
        result
        for result in results
        if result["should_answer"]
    ]

    api_found_count = sum(
        1
        for result in answerable_results
        if result["expected_api_found"]
    )

    print("\n" + "=" * 80)
    print("AUTOMATIC EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total questions: {total}")

    print(
        f"Correct answer/refusal behavior: "
        f"{behavior_correct_count}/{total} "
        f"({behavior_correct_count / total * 100:.1f}%)"
    )

    print(
        f"Expected API retrieved for answerable questions: "
        f"{api_found_count}/{len(answerable_results)} "
        f"({api_found_count / len(answerable_results) * 100:.1f}%)"
    )

    print(
        f"\nFull results saved to: {OUTPUT_FILE}"
    )
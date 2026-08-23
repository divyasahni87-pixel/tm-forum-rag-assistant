"""Measure end-to-end latency for representative TM Forum RAG questions."""

import json
import statistics
import time
from pathlib import Path

from rag_graph import rag_graph


TEST_QUESTIONS = [
    {
        "category": "definition",
        "question": "What is a BillingAccount?",
        "expected_evidence_sufficient": True,
    },
    {
        "category": "operation",
        "question": "How do I retrieve a CustomerBill by ID?",
        "expected_evidence_sufficient": True,
    },
    {
        "category": "exact_fields",
        "question": "What fields are mandatory when creating a Payment?",
        "expected_evidence_sufficient": True,
    },
    {
        "category": "cross_document",
        "question": "Which API manages customer information versus billing-account information?",
        "expected_evidence_sufficient": True,
    },
    {
        "category": "unsupported",
        "question": "What is the SLA for resolving a payment dispute?",
        "expected_evidence_sufficient": False,
    },
]

RESULTS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "latency_results.json"


def print_divider(character="="):
    """Print a consistent console divider without logging response content."""

    print(character * 80)


def main():
    """Run each representative question once and save a compact latency report."""

    results = []
    successful_latencies = []

    for index, test_case in enumerate(TEST_QUESTIONS, start=1):
        category = test_case["category"]
        question = test_case["question"]
        expected_evidence = test_case["expected_evidence_sufficient"]

        print_divider()
        print(f"LATENCY TEST {index}/{len(TEST_QUESTIONS)}")
        print(f"Category: {category}")
        print(f"Question: {question}")
        print_divider("-")

        try:
            # Time only the complete LangGraph execution, not Python startup or imports.
            start = time.perf_counter()
            result = rag_graph.invoke(
                {
                    "question": question,
                    "chunk_ids": [],
                    "context": "",
                    "evidence_sufficient": False,
                    "answer": "",
                }
            )
            elapsed = time.perf_counter() - start

            evidence_sufficient = result.get("evidence_sufficient")
            retrieved_chunk_count = len(result.get("chunk_ids", []))
            behavior_correct = evidence_sufficient == expected_evidence
            successful_latencies.append(elapsed)

            result_record = {
                "category": category,
                "question": question,
                "latency_seconds": round(elapsed, 2),
                "evidence_sufficient": evidence_sufficient,
                "expected_evidence_sufficient": expected_evidence,
                "behavior_correct": behavior_correct,
                "retrieved_chunk_count": retrieved_chunk_count,
            }

            print(f"Expected evidence: {expected_evidence}")
            print(f"Actual evidence:   {evidence_sufficient}")
            print(f"Behavior correct:  {behavior_correct}")
            print(f"Retrieved chunks: {retrieved_chunk_count}")
            print(f"Response time: {elapsed:.2f} seconds")
        except Exception as error:
            result_record = {
                "category": category,
                "question": question,
                "latency_seconds": None,
                "evidence_sufficient": None,
                "expected_evidence_sufficient": expected_evidence,
                "behavior_correct": None,
                "retrieved_chunk_count": None,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"ERROR: {result_record['error']}")

        results.append(result_record)
        print_divider()

    summary = {
        "questions_tested": len(TEST_QUESTIONS),
        "average_latency_seconds": (
            round(statistics.mean(successful_latencies), 2)
            if successful_latencies
            else None
        ),
        "minimum_latency_seconds": (
            round(min(successful_latencies), 2) if successful_latencies else None
        ),
        "maximum_latency_seconds": (
            round(max(successful_latencies), 2) if successful_latencies else None
        ),
        "median_latency_seconds": (
            round(statistics.median(successful_latencies), 2)
            if successful_latencies
            else None
        ),
    }

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY")
    print("=" * 80)
    print(f"Questions tested: {summary['questions_tested']}")
    print()
    print(f"Average latency: {summary['average_latency_seconds']:.2f} seconds" if successful_latencies else "Average latency: unavailable")
    print(f"Minimum latency: {summary['minimum_latency_seconds']:.2f} seconds" if successful_latencies else "Minimum latency: unavailable")
    print(f"Maximum latency: {summary['maximum_latency_seconds']:.2f} seconds" if successful_latencies else "Maximum latency: unavailable")
    print(f"Median latency: {summary['median_latency_seconds']:.2f} seconds" if successful_latencies else "Median latency: unavailable")
    print()

    for result_record in results:
        latency = result_record["latency_seconds"]
        latency_display = f"{latency:.2f} sec" if latency is not None else "failed"
        print(f"{result_record['category']:<16} {latency_display}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as output_file:
        json.dump({"results": results, "summary": summary}, output_file, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

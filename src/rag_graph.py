import os
from typing import TypedDict, List

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from hybrid_rerank import hybrid_rerank_search, chunk_lookup


load_dotenv()

ANSWER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)


# --------------------------------------------------
# GRAPH STATE
# --------------------------------------------------

class RAGState(TypedDict):
    question: str
    chunk_ids: List[str]
    context: str
    evidence_sufficient: bool
    answer: str


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(chunk_ids):
    context_parts = []

    for chunk_id in chunk_ids:
        chunk = chunk_lookup[chunk_id]
        metadata = chunk["metadata"]

        page = (
            metadata.get("page_label")
            or metadata.get("page")
        )
        citation = (
            f"[{metadata.get('api_id')} | "
            f"Page {page} | "
            f"Chunk {chunk_id}]"
        )

        context_parts.append(
            f"""
SOURCE
Citation: {citation}
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


# --------------------------------------------------
# NODE 1: RETRIEVE + RERANK
# --------------------------------------------------

def retrieve_node(state: RAGState):
    question = state["question"]

    chunk_ids = hybrid_rerank_search(question)
    context = build_context(chunk_ids)

    return {
        "chunk_ids": chunk_ids,
        "context": context,
    }


# --------------------------------------------------
# NODE 2: EVIDENCE CHECK
# --------------------------------------------------

def evidence_check_node(state: RAGState):

    prompt = f"""
You are checking whether retrieved TM Forum documentation
contains enough evidence to answer the user's question accurately.

USER QUESTION:
{state["question"]}

RETRIEVED CONTEXT:
{state["context"]}

RULES:

1. Use only the retrieved context.
   Do not use outside knowledge or assumptions.

2. Closely related resources are not interchangeable.
   Examples:
   - Payment is not Refund.
   - CustomerBill is not CustomerBillOnDemand.
   - Customer is not BillingAccount.
   - ProductOrder cancellation is not the same as deleting a ProductOrder.

3. For definition questions:
   The context must contain a direct description or documented purpose
   of the requested resource.

4. For operation questions:
   The requested operation, endpoint, method, or behavior must be
   explicitly supported by the context.

5. For mandatory-field questions:
   A field appearing in a JSON example does not prove it is mandatory.
   The context must explicitly identify it as mandatory.

6. For enumeration questions:
   Example values do not prove a complete enumeration.
   The complete set must be explicitly supported if the user asks
   for all possible values.

7. For comparison questions:
   The context must contain meaningful evidence about BOTH entities
   being compared.

   Do not mark the evidence sufficient merely because both entity names
   appear somewhere in the retrieved context.

   There must be enough evidence to describe each entity and make at
   least one directly supported distinction between them.

8. For relationship questions:
   Do not infer a relationship simply because two resources appear in
   the same document or example.
   The relationship must be explicitly supported.

9. For scope-boundary questions:
   Do not generalize from a related API operation to a broader business
   process unless the context explicitly supports that mapping.

10. If answering would require an important assumption or unsupported
    inference, mark the evidence insufficient.

11. Do not require every possible detail.
    The context only needs to contain enough evidence to answer the
    specific question accurately.

Return ONLY one word:

SUFFICIENT

or

INSUFFICIENT
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

    decision = (
        response.choices[0]
        .message.content
        .strip()
        .upper()
    )

    return {
        "evidence_sufficient":
            decision.startswith("SUFFICIENT")
    }


# --------------------------------------------------
# ROUTING FUNCTION
# --------------------------------------------------

def route_after_evidence_check(state: RAGState):

    if state["evidence_sufficient"]:
        return "answer"

    return "refuse"


# --------------------------------------------------
# NODE 3A: GENERATE ANSWER
# --------------------------------------------------

def answer_node(state: RAGState):

#     prompt = f"""
# You are a grounded assistant for TM Forum API documentation.

# Answer the user's question using ONLY the supplied source chunks.

# USER QUESTION:
# {state["question"]}

# SOURCE CHUNKS:
# {state["context"]}

# RULES:

# 1. Use only information supported by the source chunks.
# 2. Do not invent fields, operations, behavior, or API details.
# 3. Prefer evidence about the exact resource in the question.
# 4. CustomerBill is not CustomerBillOnDemand.
# 5. Payment is not Refund.
# 6. Keep the answer concise but useful.
# 7. Cite claims inline in this format:

# [TMFxxx | Page X | Chunk TMFxxx_XXXX]
# """
    prompt = f"""
You are a strictly grounded assistant for TM Forum API documentation.

Answer the user's question using ONLY the supplied source chunks.

USER QUESTION:
{state["question"]}

SOURCE CHUNKS:
{state["context"]}

RULES:

1. SOURCE GROUNDING

Use ONLY information explicitly supported by the supplied source chunks.

Do not use:
- outside knowledge,
- prior TM Forum knowledge,
- general telecom knowledge,
- general billing or payment knowledge,
- assumptions,
- or facts that merely seem logically correct.

If the source chunks do not support a claim, do not make that claim.


2. SOURCE-FAITHFUL WORDING

When the retrieved documentation contains a direct description of a
resource, preserve that meaning as closely as possible.

Do not strengthen, broaden, or reinterpret the source while paraphrasing.

Examples:

- If the source says "performed payment", say "performed payment".
  Do not rewrite this as "transfer of funds" unless explicitly supported.

- If the source says "amount to be refunded", say "amount to be refunded".
  Do not rewrite this as "return of funds", "reversal of funds",
  or "reverses a payment" unless explicitly supported.

- If the source says "used for billing purposes", use that wording.
  Do not rewrite this as "financial obligations", "financial entity",
  or broader financial terminology unless explicitly supported.

- If the source says a Refund has a Payment reference, state only that
  the Refund may reference a Payment.
  Do not describe the relationship as a reversal unless explicitly stated.

- If the source describes CustomerBill as a bill resource, do not call it
  "finalized", "completed", or "existing" unless that lifecycle state is
  explicitly supported.

When a direct source phrase is available, closely paraphrase or reuse it
instead of translating it into broader business terminology.


3. DO NOT INVENT OR INFER

Do not invent or infer:

- fields,
- operations,
- requirements,
- behavior,
- relationships,
- allowed values,
- lifecycle meaning,
- business meaning,
- or API details.

Do not explain what a resource "really means" beyond what the retrieved
documentation establishes.


4. EXACT RESOURCE FIRST

Prefer evidence about the EXACT resource and operation named in the question.

Closely related resources are not interchangeable.

Examples:

- CustomerBill is not CustomerBillOnDemand.
- Payment is not Refund.
- Customer is not BillingAccount.
- BillingAccount is not FinancialAccount.
- ProductOrder is not CancelProductOrder.
- Deleting a ProductOrder is not the same as requesting cancellation
  unless the documentation explicitly states that they are equivalent.

Do not use the terminology, attributes, purpose, operation, or behavior
of one resource to define another.


5. RESOURCE RELATIONSHIPS

Do not infer a relationship between resources merely because:

- they appear in the same document,
- they appear in the same JSON example,
- one contains a relatedParty,
- one references an account,
- one references another resource,
- or one API depends on another API.

An API-level dependency does not automatically establish an
entity-level relationship.

State a relationship only when the supplied chunks explicitly support it.


6. FIELDS AND REQUIREMENTS

Be precise when describing fields as:

- mandatory,
- required,
- optional,
- patchable,
- non-patchable,
- supported,
- prohibited,
- or allowed.

Only describe a field as mandatory or required when the source chunks
EXPLICITLY identify it as mandatory or required.

A field appearing in:

- a JSON request,
- a JSON response,
- a resource model,
- or a usage example

does NOT by itself mean that the field is mandatory.

Likewise, do not describe a field as optional, prohibited, supported,
or required unless the retrieved documentation explicitly supports that
level of certainty.


7. ENUMERATIONS AND EXAMPLE VALUES

Do not present example values as a complete list.

Only say:

"the supported values are..."

or equivalent wording when the supplied chunks explicitly provide a
complete enumeration.

If the chunks contain only examples, say:

"The retrieved documentation shows examples including..."

If the complete set is not available in the supplied chunks, say so.


8. OPERATIONS AND ENDPOINTS

Preserve distinctions between related operations.

For example:

DELETE /productOrder/{{id}}

and:

POST /cancelProductOrder

must not be treated as equivalent unless the documentation explicitly
states that they are.

When stating an HTTP method or endpoint, use the exact operation shown
in the retrieved source.


9. DO NOT INFER ABSENCE

Do not infer that something does not exist merely because it does not
appear in a retrieved chunk.

Do not say that a field, operation, state, relationship, or behavior:

- "does not exist",
- "is not part of",
- "is not supported",
- "cannot",
- "never",
- or equivalent

unless the supplied documentation explicitly establishes that absence.

The absence of a field from one retrieved chunk does NOT prove that the
field is absent from the complete resource model.


10. AMBIGUITY AND PARTIAL EVIDENCE

If retrieved chunks conflict or appear ambiguous, do not resolve the
ambiguity using assumptions.

State the limitation.

If the chunks support only part of the requested answer:

- answer the supported part,
- clearly state what the retrieved documentation does not establish,
- and do not fill the gap using outside knowledge.

Prefer a limited but fully grounded answer over a broader inferred answer.


11. COMPARISON QUESTIONS

When the user's question compares two resources or concepts, use this
structure:

**Entity A**
Describe Entity A using ONLY evidence specifically about Entity A.

**Entity B**
Describe Entity B using ONLY evidence specifically about Entity B.

**Key difference**
State only the directly supported distinction between the two.

For comparison questions:

- Describe both entities independently before comparing them.

- Evidence about Entity A must not be used to establish properties,
  operations, or behavior of Entity B.

- Do not infer a business relationship between the entities.

- Do not translate the source into broader business or domain terminology.

- Prefer direct definitions or documented purposes when available.

- Prefer a narrow, fully supported comparison over a broad explanation.

- If the retrieved chunks establish what each entity is but do not
  establish a relationship between them, compare their documented
  descriptions or purposes without inventing a relationship.


12. STRICT KEY-DIFFERENCE RULE

For comparison questions, the "Key difference" section must use ONLY
facts already stated and supported in the Entity A and Entity B sections.

The "Key difference" section must NOT introduce any new:

- factual claim,
- field comparison,
- relationship,
- business interpretation,
- lifecycle interpretation,
- financial meaning,
- operational behavior,
- or domain terminology.

Use the SAME source-faithful terminology already used in the Entity A
and Entity B descriptions.

Do not introduce synonyms that add new semantic meaning.

If the evidence supports only a limited distinction, state only that
limited distinction.


13. CITATIONS

Cite factual claims inline using this exact format:

[TMFxxx | Page X | Chunk TMFxxx_XXXX]

Every important API-specific claim must have an inline citation.

Every citation must support the EXACT factual claim immediately before it.

Do not cite a chunk merely because it discusses a related resource.


14. ENDPOINT CITATION VALIDATION

If a claim contains an HTTP method or endpoint, such as:

GET /customerBill/{{id}}
POST /refund
DELETE /productOrder/{{id}}

the cited chunk must explicitly contain that same operation.

Never use evidence about one resource's endpoint to support another
resource's endpoint.

For example:

A CustomerBillOnDemand chunk must not support a claim about
GET /customerBill/{{id}} unless that exact CustomerBill operation
also appears in that chunk.


15. CLAIM-TO-CITATION ALIGNMENT

Do not attach one citation to a sentence containing several factual
claims unless the cited chunk supports ALL of those claims.

Split the sentence and cite claims separately when necessary.

If no retrieved chunk directly supports a claim:

- remove the claim,
- weaken it to match the evidence,
- or explicitly state that the retrieved documentation does not
  establish it.

Never attach a merely related citation to an unsupported claim.


16. ANSWER STYLE

Keep the answer concise, clear, and useful.

Prefer the most directly relevant evidence instead of summarizing every
retrieved chunk.

For comparison questions, normally use:

**Entity A**
One concise, directly supported paragraph with citation(s).

**Entity B**
One concise, directly supported paragraph with citation(s).

**Key difference**
One short comparison using only facts already established above.

Do not add a broader "In summary" interpretation after the supported
comparison.

Do not add extra attributes unless they are useful for answering the
specific comparison.

For non-comparison questions, answer naturally and concisely.

If only a limited answer is supported, provide that limited answer rather
than expanding beyond the evidence.


17. FINAL GROUNDING CHECK

Before returning the answer, silently verify EACH factual claim:

- Is this claim explicitly supported by a supplied source chunk?

- Does the citation immediately following it support that exact claim?

- Did I make the source broader, stronger, or more certain than it is?

- Did I replace source terminology with broader domain terminology?

- Did I infer a relationship that the source does not explicitly state?

- Did I introduce telecom, billing, payment, or TM Forum knowledge that
  is absent from the supplied context?

- Did I make an absence claim without explicit evidence?

- If this is a comparison, did the Key difference section introduce
  anything that was not already established in the Entity A and Entity B
  sections?

If any claim fails this check:

REMOVE IT,
WEAKEN IT TO MATCH THE SOURCE,
or STATE THAT THE RETRIEVED DOCUMENTATION DOES NOT ESTABLISH IT.

Do not mention this verification process in the final answer.
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

    return {
        "answer":
            response.choices[0]
            .message.content
            .strip()
    }


# --------------------------------------------------
# NODE 3B: REFUSE
# --------------------------------------------------

def refuse_node(state: RAGState):

    return {
        "answer":
            "I could not find enough information "
            "in the retrieved TM Forum documentation."
    }


# --------------------------------------------------
# BUILD LANGGRAPH
# --------------------------------------------------

builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("evidence_check", evidence_check_node)
builder.add_node("answer", answer_node)
builder.add_node("refuse", refuse_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "evidence_check")

builder.add_conditional_edges(
    "evidence_check",
    route_after_evidence_check,
    {
        "answer": "answer",
        "refuse": "refuse",
    },
)

builder.add_edge("answer", END)
builder.add_edge("refuse", END)

rag_graph = builder.compile()


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    questions = [
        # "What is a BillingAccount?",
        # "How do I retrieve a customer bill?",
        # "What fields are mandatory when creating a payment?",
        # "What is the SLA for resolving a payment dispute?",
        # "What is a ProductOrder?",
        # "How do I cancel a ProductOrder?",
        # "What actions can be performed on a ProductOrderItem?",
       # "What is a Customer?",
        #"How do I update a customer's contact information?",
        #"Which API manages customer information versus billing-account information?",
        "What is the difference between a BillingAccount and a Customer?",
        "What is the difference between a Payment and a Refund?",
        "What is the difference between CustomerBill and CustomerBillOnDemand?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        result = rag_graph.invoke(
            {
                "question": question,
                "chunk_ids": [],
                "context": "",
                "evidence_sufficient": False,
                "answer": "",
            }
        )

        print("\nEVIDENCE SUFFICIENT:")
        print(result["evidence_sufficient"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        print(", ".join(result["chunk_ids"]))

import re

import streamlit as st

from hybrid_rerank import chunk_lookup
from rag_graph import rag_graph


st.set_page_config(
    page_title="TM Forum RAG Assistant",
    page_icon=":material/cell_tower:",
    layout="wide",
)


# Centralized visual polish: these rules only refine spacing, borders, and
# hierarchy around native Streamlit components; no application logic lives here.
st.html(
    """
    <style>
      /* Theme */
      :root {
        --navy: #0B1628;
        --navy-soft: #111F35;
        --slate: #1E293B;
        --blue: #3B82F6;
        --blue-soft: #60A5FA;
        --canvas: #F7F9FC;
        --card: #FFFFFF;
        --secondary: #F1F5F9;
        --ink: #172033;
        --muted: #64748B;
        --border: #DCE3EC;
      }
      [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
      }
      .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 7rem;
      }

      /* Sidebar */
      [data-testid="stSidebar"] > div:first-child {
        background: #17243A;
      }
      [data-testid="stSidebar"] .sidebar-section-title {
        color: #F8FAFC;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-top: 12px;
        margin-bottom: 10px;
      }
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B7C4D6;
      }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 13px;
      }
      [data-testid="stSidebar"] .knowledge-base-row {
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: grid;
        gap: 0.4rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        padding: 8px 4px;
      }
      [data-testid="stSidebar"] .knowledge-base-api-id {
        color: #7DB4FF;
        font-size: 13px;
        font-weight: 700;
      }
      [data-testid="stSidebar"] .knowledge-base-api-name {
        color: #F1F5F9;
        font-size: 13px;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      [data-testid="stSidebar"] .knowledge-base-metadata {
        color: #AEBED0;
        font-size: 12px;
        white-space: nowrap;
      }
      [data-testid="stSidebar"] .knowledge-base-total {
        color: #B7C4D6;
        font-size: 12.5px;
        padding: 8px 4px 0;
      }
      [class*="st-key-pipeline-step-"] {
        background: transparent;
        border: 0;
        border-left: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 0;
        margin-left: 0.25rem;
        padding: 0.05rem 0 0.05rem 0.7rem;
      }
      [class*="st-key-pipeline-step-"] p {
        color: #F1F5F9 !important;
        font-size: 14px;
        font-weight: 600;
      }
      [class*="st-key-pipeline-step-"] p strong {
        color: #60A5FA;
      }
      [class*="st-key-pipeline-step-"] [data-testid="stCaptionContainer"] {
        font-size: 12.75px;
        color: #AEBED0;
      }
      [data-testid="stSidebar"] .stButton button {
        background: #213149;
        border: 1px solid #3B4E69;
        border-radius: 8px;
        color: #E8EEF7;
        font-size: 13.75px;
        font-weight: 500;
        text-align: left;
      }
      [data-testid="stSidebar"] .stButton button:hover {
        background: #293D59;
        border-color: #60A5FA;
        color: #FFFFFF;
      }

      /* Header */
      [class*="st-key-hero"] {
        border-left: 3px solid var(--blue);
        padding: 0.15rem 0 0.6rem 1rem;
      }
      [class*="st-key-hero"] h1 {
        color: var(--navy);
        font-size: clamp(2rem, 4vw, 2.65rem);
        letter-spacing: -0.035em;
        margin-bottom: 0.25rem;
      }
      [class*="st-key-hero"] h3 {
        color: var(--slate);
        font-size: 1.05rem;
        font-weight: 600;
      }
      [class*="st-key-hero"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
      }

      /* Project insights */
      [class*="st-key-insight-metric-"] {
        background: #F8FAFC;
        border-color: var(--border);
        border-radius: 9px;
        min-height: 82px;
      }
      [class*="st-key-insight-metric-"] [data-testid="stMetricValue"] {
        color: var(--blue);
        font-size: 1.4rem;
        font-weight: 700;
      }
      [class*="st-key-insight-metric-"] [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.78rem;
      }
      [class*="st-key-project-insights-notes"] [data-testid="stCaptionContainer"] {
        color: #3F4D63 !important;
      }
      [class*="st-key-project-insights-coverage-note"] [data-testid="stCaptionContainer"] {
        color: #3F4D63 !important;
      }
      .coverage-list {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
      }
      .coverage-label {
        display: grid;
        align-items: baseline;
        gap: 0.45rem;
        grid-template-columns: auto minmax(0, 1fr) auto;
        margin-bottom: 0.25rem;
      }
      .coverage-api {
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 700;
      }
      .coverage-name {
        color: var(--muted);
        font-size: 0.76rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .coverage-count {
        color: var(--ink);
        font-size: 0.76rem;
        font-variant-numeric: tabular-nums;
        font-weight: 600;
      }
      .coverage-track {
        background: #E8EEF5;
        border-radius: 999px;
        height: 7px;
        overflow: hidden;
      }
      .coverage-fill {
        background: #3B82F6;
        border-radius: inherit;
        height: 100%;
      }
      .architecture-flow {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.3rem;
        color: var(--ink);
        font-size: 0.78rem;
        text-align: center;
      }
      .architecture-node,
      .architecture-branch,
      .architecture-outcome {
        background: #FFFFFF;
        border: 1px solid #CFE0F7;
        border-radius: 7px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        padding: 0.35rem 0.6rem;
        font-weight: 600;
      }
      .architecture-node--accent {
        background: #EFF6FF;
        border-color: #93C5FD;
        color: #1D4ED8;
      }
      .architecture-node small,
      .architecture-branch small,
      .architecture-outcome small {
        display: block;
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 500;
        margin-top: 0.1rem;
      }
      .architecture-arrow {
        color: #3B82F6;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1;
      }
      .architecture-split,
      .architecture-outcomes {
        display: grid;
        gap: 0.5rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        width: min(100%, 360px);
      }
      .architecture-comparison {
        background: #F8FAFC;
        border: 1px dashed #93C5FD;
        border-radius: 7px;
        color: #31506E;
        font-size: 0.72rem;
        padding: 0.35rem 0.5rem;
        width: min(100%, 360px);
      }
      .architecture-comparison-title {
        color: #1D4ED8;
        font-weight: 700;
        margin-bottom: 0.25rem;
      }
      .architecture-comparison-flow {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        justify-content: center;
      }
      .architecture-comparison-step {
        background: #FFFFFF;
        border: 1px solid #CFE0F7;
        border-radius: 5px;
        padding: 0.2rem 0.35rem;
      }
      .architecture-comparison-arrow {
        color: #3B82F6;
        font-weight: 700;
      }
      .architecture-outcome--success {
        background: #F0FDF4;
        border-color: #BBF7D0;
      }
      .architecture-outcome--warning {
        background: #FFF7ED;
        border-color: #FED7AA;
      }
      .incremental-flow {
        color: #31506E;
        font-size: 0.8rem;
        font-weight: 600;
        line-height: 1.65;
      }

      /* Chat */
      [data-testid="stChatMessage"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.8rem;
      }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #EEF3F8;
        box-shadow: none;
      }
      [data-testid="stChatMessage"] p,
      [data-testid="stChatMessage"] li {
        color: var(--ink);
        line-height: 1.68;
      }
      [data-testid="stChatMessage"] p {
        margin-bottom: 0.7rem;
      }
      [data-testid="stChatMessage"] code {
        background: var(--secondary);
        color: var(--navy);
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 0.1rem 0.25rem;
      }
      [data-testid="stChatInput"] textarea {
        background: var(--card);
        border: 0;
        color: #172033;
      }
      [data-testid="stChatInput"] {
        background: #FFFFFF;
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
      }
      [data-testid="stChatInput"]:focus-within {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.12);
      }
      [data-testid="stChatInput"] textarea::placeholder {
        color: #7C8A9E;
      }
      [data-testid="stChatInput"] button {
        background: #E8F0FE;
        color: #2563EB;
      }
      [data-testid="stChatInput"] button:hover {
        background: #DBEAFE;
      }

      /* Sources */
      [data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 9px;
      }
      [data-testid="stExpander"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
      }
    </style>
    """
)


# Short, presentation-friendly labels map to the same questions used before.
EXAMPLE_PROMPTS = [
    ("Define ProductOrder", "What is a ProductOrder?"),
    ("Cancel ProductOrder", "How do I cancel a ProductOrder?"),
    ("Define Customer", "What is a Customer?"),
    ("Update Contact Info", "How do I update a customer's contact information?"),
    (
        "Customer vs Billing Account",
        "Which API manages customer information versus billing-account information?",
    ),
    ("Required Payment Fields", "What fields are mandatory when creating a Payment?"),
]


# Match only the citation format emitted by the RAG backend. Fenced code blocks
# are split out before formatting so JSON and other examples remain unchanged.
CITATION_TOKEN = r"\[TMF\d{3} \| Page [^\]\n]+ \| Chunk TMF\d{3}_\d{4}\]"
CITATION_PATTERN = re.compile(f"({CITATION_TOKEN})")
CITATION_CLUSTER_PATTERN = re.compile(
    rf"(?P<cluster>{CITATION_TOKEN}(?:[,\s]*{CITATION_TOKEN})*)"
    r"(?P<terminal>[.!?])?"
)
CODE_FENCE_PATTERN = re.compile(r"(```.*?```)", re.DOTALL)


def format_text_segment_with_citations(text):
    """Format citations in non-code text without creating orphaned punctuation."""

    blocks = []
    cursor = 0

    for match in CITATION_CLUSTER_PATTERN.finditer(text):
        claim = text[cursor:match.start()].strip()
        terminal = match.group("terminal")

        # A period, question mark, or exclamation mark written after a citation
        # belongs to the preceding claim, not on a separate line after it.
        if terminal and claim:
            if claim[-1] not in ".?!":
                claim += terminal

        # Preserve commas, semicolons, and colons with adjacent prose instead
        # of allowing a punctuation-only block to be created.
        if claim:
            if claim[0] in ",;:" and blocks:
                blocks[-1] += claim
            else:
                blocks.append(claim)

        citations = "\n".join(CITATION_PATTERN.findall(match.group("cluster")))
        if terminal and not claim:
            citations += terminal
        blocks.append(citations)
        cursor = match.end()

    remaining_text = text[cursor:].strip()
    if remaining_text:
        if remaining_text[0] in ",;:" and blocks:
            blocks[-1] += remaining_text
        else:
            blocks.append(remaining_text)

    return "\n\n".join(blocks)


def format_answer_for_display(answer):
    """Place backend citations on separate Markdown lines without changing them."""

    formatted_parts = []

    for index, part in enumerate(CODE_FENCE_PATTERN.split(answer)):
        if index % 2:
            formatted_parts.append(part)
        else:
            formatted_parts.append(format_text_segment_with_citations(part))

    return "".join(formatted_parts).strip()


def initialize_session_state():
    """Create per-browser-session chat state before rendering the UI."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)


def clear_conversation():
    """Clear only the current browser session's chat history."""

    st.session_state.messages = []
    st.session_state.pending_question = None


def source_details(chunk_ids):
    """Render source metadata for graph-retrieved chunk IDs."""

    with st.expander("Sources & evidence", icon=":material/source:"):
        if not chunk_ids:
            st.caption("No source chunks were returned for this response.")
            return

        for chunk_id in chunk_ids:
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                st.caption(f"{chunk_id}: metadata is unavailable locally.")
                continue

            metadata = chunk["metadata"]
            page = metadata.get("page_label", metadata.get("page"))

            with st.container(border=True):
                st.markdown(
                    f"**{metadata.get('api_id')} · {metadata.get('api_name')}**"
                )
                st.caption(
                    f"Version {metadata.get('version')} · "
                    f"Page {page} · Chunk {chunk_id}"
                )
                st.badge(metadata.get("content_type", "source"), color="gray")
                with st.expander("View retrieved text"):
                    st.markdown(chunk["text"])


def render_assistant_message(message):
    """Display a saved answer without invoking the RAG graph again."""

    st.markdown(format_answer_for_display(message["answer"]))

    if message.get("evidence_sufficient"):
        st.badge(
            "Grounded in TM Forum documentation",
            icon=":material/check_circle:",
            color="green",
        )
    else:
        st.badge(
            "Insufficient evidence in current knowledge base",
            icon=":material/warning:",
            color="orange",
        )

    source_details(message.get("chunk_ids", []))

    if message.get("error_details"):
        with st.expander("Developer / Retrieval Details", icon=":material/bug_report:"):
            st.code(message["error_details"])


initialize_session_state()


with st.sidebar:
    st.markdown(
        '<div class="sidebar-section-title">Knowledge base</div>',
        unsafe_allow_html=True,
    )
    knowledge_base_apis = [
        ("TMF666", "Account Management", "v5.0.0", "370 chunks"),
        ("TMF678", "Customer Bill Management", "v5.0.0", "102 chunks"),
        ("TMF676", "Payment Management", "v4.0.0", "66 chunks"),
        ("TMF622", "Product Ordering Management", "v5.0.0", "299 chunks"),
        ("TMF629", "Customer Management", "v5.0.0", "65 chunks"),
    ]
    knowledge_base_rows = "".join(
        f"""
        <div class="knowledge-base-row">
          <span class="knowledge-base-api-id">{api_id}</span>
          <span class="knowledge-base-api-name">{api_name}</span>
          <span class="knowledge-base-metadata">{version} &middot; {chunk_count}</span>
        </div>
        """
        for api_id, api_name, version, chunk_count in knowledge_base_apis
    )
    st.html(
        f"""
        <div class="knowledge-base-list">
          {knowledge_base_rows}
          <div class="knowledge-base-total">&bull; 902 indexed chunks</div>
        </div>
        """
    )

    st.space("small")
    st.markdown(
        '<div class="sidebar-section-title">Retrieval pipeline</div>',
        unsafe_allow_html=True,
    )
    pipeline_steps = [
        ("01", "Hybrid retrieval", "Dense + BM25"),
        ("02", "Rank fusion", "Reciprocal Rank Fusion"),
        ("03", "LLM reranking", "Relevance refinement"),
        ("04", "Evidence check", "LangGraph validation"),
        ("05", "Answer", "Grounded response or refusal"),
    ]
    for step, title, detail in pipeline_steps:
        with st.container(border=True, key=f"pipeline-step-{step}"):
            st.markdown(f"**{step}  {title}**")
            st.caption(detail)

    st.space("small")
    st.markdown(
        '<div class="sidebar-section-title">Try asking</div>',
        unsafe_allow_html=True,
    )
    for index, (label, question) in enumerate(EXAMPLE_PROMPTS):
        if st.button(label, key=f"example_{index}", width="stretch"):
            st.session_state.pending_question = question

    st.space("medium")
    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        width="stretch",
        type="secondary",
    ):
        clear_conversation()
        st.rerun()

    if st.session_state.messages:
        latest_assistant = next(
            (
                message
                for message in reversed(st.session_state.messages)
                if message["role"] == "assistant"
            ),
            None,
        )
        if latest_assistant:
            with st.expander(
                "Developer / Retrieval Details",
                icon=":material/analytics:",
            ):
                st.write(
                    "Evidence sufficient:",
                    latest_assistant.get("evidence_sufficient", False),
                )
                st.write(
                    "Retrieved chunks:",
                    len(latest_assistant.get("chunk_ids", [])),
                )
                st.code(
                    "\n".join(latest_assistant.get("chunk_ids", []))
                    or "No chunk IDs returned."
                )

    st.space("medium")
    st.caption("Built with LangGraph · Pinecone · Nebius")


with st.container(key="hero"):
    st.badge(
        "TELECOM KNOWLEDGE ASSISTANT",
        icon=":material/cell_tower:",
        color="blue",
    )
    st.title("TM Forum RAG Assistant")
    st.subheader(
        "Grounded answers across TM Forum Customer, Product Ordering, Account, Billing, and Payment APIs.",
        divider=False,
    )
    st.caption("Hybrid retrieval • LLM reranking • LangGraph evidence validation")


with st.expander("Project Insights", icon=":material/insights:"):
    insight_left, insight_right = st.columns((1, 1.15), gap="medium")

    with insight_left:
        st.subheader("Knowledge Base Coverage", divider=False)
        coverage_rows = [
            ("TMF666", "Account Management", 370),
            ("TMF622", "Product Ordering Management", 299),
            ("TMF678", "Customer Bill Management", 102),
            ("TMF676", "Payment Management", 66),
            ("TMF629", "Customer Management", 65),
        ]
        largest_coverage = coverage_rows[0][2]
        coverage_html = "".join(
            f"""
            <div class="coverage-row">
              <div class="coverage-label">
                <span class="coverage-api">{api_id}</span>
                <span class="coverage-name">{api_name}</span>
                <span class="coverage-count">{chunks}</span>
              </div>
              <div class="coverage-track">
                <div class="coverage-fill" style="width: {chunks / largest_coverage * 100:.1f}%"></div>
              </div>
            </div>
            """
            for api_id, api_name, chunks in coverage_rows
        )
        st.html(f'<div class="coverage-list">{coverage_html}</div>')
        with st.container(key="project-insights-coverage-note"):
            st.caption("902 indexed chunks across five TM Forum APIs.")

    with insight_right:
        st.subheader("RAG Architecture", divider=False)
        st.html(
            """
            <div class="architecture-flow">
              <div class="architecture-node">User question</div>
              <div class="architecture-arrow">&darr;</div>
              <div class="architecture-node architecture-node--accent">Hybrid retrieval</div>
              <div class="architecture-comparison">
                <div class="architecture-comparison-title">Comparison-aware branch <small>when a comparison is detected</small></div>
                <div class="architecture-comparison-flow">
                  <span class="architecture-comparison-step">Comparison detected</span>
                  <span class="architecture-comparison-arrow">&rarr;</span>
                  <span class="architecture-comparison-step">Entity A focused retrieval</span>
                  <span class="architecture-comparison-arrow">+</span>
                  <span class="architecture-comparison-step">Entity B focused retrieval</span>
                  <span class="architecture-comparison-arrow">&rarr;</span>
                  <span class="architecture-comparison-step">Balanced evidence</span>
                </div>
              </div>
              <div class="architecture-split">
                <div class="architecture-branch">Dense search<small>Pinecone</small></div>
                <div class="architecture-branch">BM25 search<small>Keyword</small></div>
              </div>
              <div class="architecture-arrow">&darr;</div>
              <div class="architecture-node">RRF fusion</div>
              <div class="architecture-arrow">&darr;</div>
              <div class="architecture-node">LLM reranking</div>
              <div class="architecture-arrow">&darr;</div>
              <div class="architecture-node architecture-node--accent">LangGraph evidence check</div>
              <div class="architecture-outcomes">
                <div class="architecture-outcome architecture-outcome--success">Evidence found<small>Grounded answer &rarr; Citations</small></div>
                <div class="architecture-outcome architecture-outcome--warning">Insufficient evidence<small>Refusal</small></div>
              </div>
            </div>
            """
        )

    st.subheader("Evaluation Summary", divider=False)
    evaluation_columns = st.columns(4, gap="small")
    for column, value, label in zip(
        evaluation_columns,
        ["25", "96%", "Improved", "Grounded"],
        [
            "Evaluation questions",
            "Automatic behavior",
            "Comparison retrieval",
            "Refusal protection",
        ],
    ):
        with column:
            with st.container(
                border=True,
                key=f"insight-metric-{label.lower().replace(' ', '-')}",
            ):
                st.metric(label, value)

    with st.container(key="project-insights-notes"):
        st.caption(
            "25 evaluation questions across definitions, operations, exact fields, cross-document reasoning, comparisons, and unsupported queries."
        )
        st.caption("24 / 25 questions showed correct automatic answer/refusal behavior (96%).")
        st.caption(
            "Evaluation-driven improvement: Multi-entity comparison failures led to comparison-aware retrieval that preserves evidence for both entities before reranking."
        )
        st.caption(
            "Remaining limitation: Scope-boundary questions can still confuse a related API operation with the broader business process being asked about."
        )

        st.markdown("**Built incrementally**")
        st.html(
            """
            <div class="incremental-flow">
              Dense &rarr; BM25 &rarr; Hybrid + RRF &rarr; LLM Reranking &rarr; Evidence Gate &rarr; Comparison-Aware Retrieval
            </div>
            """
        )
        st.caption("Each layer was added after evaluation exposed a specific failure mode.")


for message in st.session_state.messages:
    avatar = ":material/person:" if message["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message)


typed_question = st.chat_input(
    "Ask about customers, product orders, accounts, bills, payments, or refunds...",
    submit_mode="disable",
)
question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None


if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        try:
            with st.spinner("Searching and validating TM Forum documentation..."):
                result = rag_graph.invoke(
                    {
                        "question": question,
                        "chunk_ids": [],
                        "context": "",
                        "evidence_sufficient": False,
                        "answer": "",
                    }
                )

            assistant_message = {
                "role": "assistant",
                "answer": result["answer"],
                "chunk_ids": result["chunk_ids"],
                "evidence_sufficient": result["evidence_sufficient"],
            }
        except Exception as error:
            assistant_message = {
                "role": "assistant",
                "answer": "Unable to process the question right now. Please try again.",
                "chunk_ids": [],
                "evidence_sufficient": False,
                "error_details": f"{type(error).__name__}: {error}",
            }

        render_assistant_message(assistant_message)
        st.session_state.messages.append(assistant_message)

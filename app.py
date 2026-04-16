from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from feedback.logger import clear_feedback_log, get_feedback_stats, log_feedback
from llm.responder import build_answer_response
from memory.embedder import embed_text
from memory.vector_store import VectorStore
from retriever.semantic_search import search_memory
from utils.paths import EMBEDDING_DIM, MEMORY_INDEX_PATH, MEMORY_METADATA_PATH
from utils.text_utils import normalize_text


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(
        dim=EMBEDDING_DIM,
        index_path=MEMORY_INDEX_PATH,
        metadata_path=MEMORY_METADATA_PATH,
    )


def clear_memory_editor_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("memory_editor_"):
            del st.session_state[key]


def reset_answer_state() -> None:
    st.session_state.last_question = None
    st.session_state.last_results = None


def reset_app_state() -> None:
    reset_answer_state()
    clear_memory_editor_state()


def format_confidence_score(score: Optional[int]) -> str:
    if score is None:
        return "n/a"
    return f"{score}/100"


def format_distance(distance: Optional[float]) -> str:
    if distance is None:
        return "n/a"
    return f"{distance:.4f}"


def render_response(question: str, response: Dict[str, Any]) -> None:
    st.subheader("Latest Answer")
    st.caption(f"Question: {question}")

    if response["found"]:
        st.success(response["answer"])
    else:
        st.warning(response["answer"])

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Confidence score", format_confidence_score(response.get("confidence_score")))
    metric_col2.metric(
        "Confidence tier",
        str(response.get("confidence_label") or "n/a").title(),
    )
    metric_col3.metric("Semantic distance", format_distance(response.get("distance")))

    if response.get("distance_gap") is not None:
        st.caption(f"Gap to second-best match: {response['distance_gap']:.4f}")

    if response.get("source_text"):
        st.markdown(
            f"**{'Source memory' if response['found'] else 'Closest stored memory'}**"
        )
        st.write(response["source_text"])

    if response.get("alternate_source_text") and response.get("rejection_reason") == "ambiguous":
        st.markdown("**Another similarly close memory**")
        st.write(response["alternate_source_text"])

    if response.get("rejection_reason") == "ambiguous":
        st.info("The top two memories were too close together, so the app refused to guess.")
    elif response.get("rejection_reason") == "low_confidence" and response.get("source_text"):
        st.info("A related memory was found, but the match strength was below the current cutoff.")

    if response.get("confidence_score") is not None:
        st.caption(
            "Confidence score is a retrieval heuristic based on match distance and separation from the second-best result."
        )


st.set_page_config(page_title="AI Learning Companion", layout="wide")
st.title("AI Learning Companion")
st.caption(
    "Teach facts, ask questions in different wording, and see whether semantic retrieval finds a confident answer."
)

if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None

store = get_store()

with st.sidebar:
    st.header("Settings")
    max_distance = st.slider(
        "Confidence cutoff (lower is stricter)",
        min_value=0.10,
        max_value=2.00,
        value=1.00,
        step=0.05,
        help="If the best match is farther than this, the app will say it does not know.",
    )
    min_distance_gap = st.slider(
        "Ambiguity buffer",
        min_value=0.00,
        max_value=0.50,
        value=0.15,
        step=0.01,
        help="If the best and second-best matches are too close together, the app will refuse to guess.",
    )

    st.header("Project Snapshot")
    st.metric("Facts in memory", store.size())
    feedback_stats = get_feedback_stats()
    st.metric("Feedback entries", feedback_stats["total"])

    st.header("Controls")
    if st.button("Clear Memory", type="secondary", use_container_width=True):
        store.clear()
        reset_app_state()
        st.success("Memory cleared.")
        st.rerun()

    if st.button("Clear Feedback Log", type="secondary", use_container_width=True):
        clear_feedback_log()
        st.success("Feedback log cleared.")
        st.rerun()

current_response: Dict[str, Any] | None = None
if st.session_state.last_question and st.session_state.last_results is not None:
    current_response = build_answer_response(
        st.session_state.last_results,
        max_distance=max_distance,
        min_distance_gap=min_distance_gap,
    )

teach_col, ask_col = st.columns(2, gap="large")

with teach_col:
    st.subheader("Teach Me Something")
    with st.form(key="teach_form", clear_on_submit=True):
        fact_input = st.text_area(
            "Enter a fact to remember:",
            placeholder="Example: My favorite music is rock.",
            height=140,
        )
        teach_submit = st.form_submit_button("Teach")

        if teach_submit:
            cleaned_fact = normalize_text(fact_input)
            if not cleaned_fact:
                st.warning("Please enter a fact before submitting.")
            else:
                added = store.add(embed_text(cleaned_fact), cleaned_fact)
                if added:
                    st.success("Learned! The fact was added to memory.")
                else:
                    st.info("That fact is already in memory.")
                st.rerun()

    facts = store.facts()

    with st.expander("View Stored Facts"):
        if facts:
            for i, fact in enumerate(facts, start=1):
                st.write(f"{i}. {fact}")
        else:
            st.info("No facts stored yet.")

    with st.expander("Manage Stored Facts"):
        if not facts:
            st.info("Teach at least one fact before editing or deleting memory.")
        else:
            selected_index = st.selectbox(
                "Select a stored fact",
                options=list(range(len(facts))),
                format_func=lambda index: facts[index],
                key="memory_editor_selection",
            )
            selected_fact = facts[selected_index]
            edited_fact = st.text_area(
                "Edit selected fact",
                value=selected_fact,
                height=120,
                key=f"memory_editor_text_{selected_index}",
            )

            manage_col1, manage_col2 = st.columns(2)
            with manage_col1:
                if st.button("Save Changes", use_container_width=True, key="memory_editor_save"):
                    try:
                        status = store.update_fact(selected_fact, edited_fact, embed_text)
                    except ValueError:
                        st.warning("Please enter a non-empty fact before saving.")
                    else:
                        if status == "updated":
                            reset_app_state()
                            st.success("Stored fact updated.")
                            st.rerun()
                        elif status == "duplicate":
                            st.info("That updated fact already exists in memory.")
                        else:
                            st.warning("The selected fact could not be found. Try again.")

            with manage_col2:
                if st.button(
                    "Delete Selected Fact",
                    use_container_width=True,
                    key="memory_editor_delete",
                ):
                    deleted = store.delete_fact(selected_fact, embed_text)
                    if deleted:
                        reset_app_state()
                        st.success("Stored fact deleted.")
                        st.rerun()
                    else:
                        st.warning("The selected fact could not be found. Try again.")

with ask_col:
    st.subheader("Ask a Question")
    with st.form(key="ask_form"):
        question = st.text_input(
            "What would you like to know?",
            placeholder="Example: What is my favorite music?",
        )
        ask_submit = st.form_submit_button("Ask")

        if ask_submit:
            cleaned_question = normalize_text(question)
            if not cleaned_question:
                st.warning("Please enter a question before submitting.")
            else:
                results = search_memory(cleaned_question, store, top_k=2)
                st.session_state.last_question = cleaned_question
                st.session_state.last_results = results
                st.rerun()

    if st.session_state.last_question and current_response:
        render_response(st.session_state.last_question, current_response)

if st.session_state.last_question and current_response:
    st.divider()
    st.subheader("Answer Feedback")
    st.write(f"Question: {st.session_state.last_question}")
    st.write(f"Answer shown: {current_response['answer']}")

    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        if st.button("Correct / Helpful", use_container_width=True):
            log_feedback(
                question=st.session_state.last_question,
                answer=current_response["answer"],
                distance=current_response.get("distance"),
                source_text=current_response.get("source_text"),
                confidence_score=current_response.get("confidence_score"),
                rejection_reason=current_response.get("rejection_reason"),
                label="up",
            )
            st.success("Feedback saved.")
            st.rerun()

    with feedback_col2:
        if st.button("Incorrect / Unhelpful", use_container_width=True):
            log_feedback(
                question=st.session_state.last_question,
                answer=current_response["answer"],
                distance=current_response.get("distance"),
                source_text=current_response.get("source_text"),
                confidence_score=current_response.get("confidence_score"),
                rejection_reason=current_response.get("rejection_reason"),
                label="down",
            )
            st.success("Feedback saved.")
            st.rerun()

    refreshed_stats = get_feedback_stats()
    if refreshed_stats["total"] > 0:
        st.caption(
            f"Feedback logged: {refreshed_stats['total']} total | "
            f"{refreshed_stats['up']} positive | {refreshed_stats['down']} negative"
        )

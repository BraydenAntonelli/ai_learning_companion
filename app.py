from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, Optional

import streamlit as st

from feedback.logger import clear_feedback_log, get_feedback_stats, log_feedback
from llm.responder import build_answer_response, build_teach_response, format_category_label
from memory.classifier import CATEGORY_OPTIONS, detect_message_intent, strip_teach_prefix
from memory.embedder import embed_text, embed_texts
from memory.models import MemoryDraft, MemoryRecord
from memory.vector_store import VectorStore
from retriever.semantic_search import search_memory
from utils.document_utils import build_document_ingestion_plan
from utils.paths import (
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_DIM,
    MEMORY_INDEX_PATH,
    MEMORY_METADATA_PATH,
    SUPPORTED_UPLOAD_EXTENSIONS,
)
from utils.study_utils import build_flashcard
from utils.text_utils import normalize_text, split_tags, truncate_text


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore(
        dim=EMBEDDING_DIM,
        index_path=MEMORY_INDEX_PATH,
        metadata_path=MEMORY_METADATA_PATH,
        embed_fn=embed_text,
        embed_batch_fn=embed_texts,
    )


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "chat_history": [],
        "last_question": None,
        "last_response": None,
        "study_record_id": None,
        "study_revealed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_memory_editor_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("memory_editor_"):
            del st.session_state[key]


def reset_answer_state() -> None:
    st.session_state.last_question = None
    st.session_state.last_response = None


def reset_app_state() -> None:
    reset_answer_state()
    clear_memory_editor_state()
    st.session_state.study_record_id = None
    st.session_state.study_revealed = False


def append_chat_message(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def format_similarity(score: Optional[float]) -> str:
    if score is None:
        return "n/a"
    return f"{score:.3f}"


def format_confidence(score: Optional[int]) -> str:
    if score is None:
        return "n/a"
    return f"{score}/100"


def format_source_label(source: str) -> str:
    if source.startswith("upload:"):
        return source.removeprefix("upload:")
    return source.replace("_", " ").title()


def render_response(response: Dict[str, Any]) -> None:
    st.subheader("Latest Answer")
    if response["found"]:
        st.success(response["answer"])
    else:
        st.warning(response["answer"])

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Cosine similarity", format_similarity(response.get("score")))
    metric_col2.metric("Confidence score", format_confidence(response.get("confidence_score")))
    metric_col3.metric(
        "Confidence tier",
        str(response.get("confidence_label") or "n/a").title(),
    )

    if response.get("score_gap") is not None:
        st.caption(f"Gap to second-best match: {response['score_gap']:.3f}")

    source_record: Optional[MemoryRecord] = response.get("source_record")
    if source_record is not None:
        heading = "Source memory" if response["found"] else "Closest stored memory"
        st.markdown(f"**{heading}**")
        st.write(source_record.text)
        st.caption(
            f"Category: {format_category_label(source_record.category)} | "
            f"Source: {format_source_label(source_record.source)}"
        )
        if source_record.tags:
            st.caption(f"Tags: {', '.join(source_record.tags)}")

    alternate_source: Optional[MemoryRecord] = response.get("alternate_source_record")
    if alternate_source is not None and response.get("rejection_reason") == "ambiguous":
        st.markdown("**Another similarly close memory**")
        st.write(alternate_source.text)
        st.caption(
            f"Category: {format_category_label(alternate_source.category)} | "
            f"Source: {format_source_label(alternate_source.source)}"
        )

    if response.get("rejection_reason") == "ambiguous":
        st.info("The top two memories were too close together, so the app refused to guess.")
    elif response.get("rejection_reason") == "low_confidence" and source_record is not None:
        st.info("A related memory was found, but the similarity score was below the current cutoff.")


def render_feedback_section(question: str, response: Dict[str, Any]) -> None:
    st.divider()
    st.subheader("Answer Feedback")
    st.write(f"Question: {question}")
    st.write(f"Answer shown: {response['answer']}")

    source_record: Optional[MemoryRecord] = response.get("source_record")

    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        if st.button("Correct / Helpful", use_container_width=True, key="feedback_up"):
            log_feedback(
                question=question,
                answer=response["answer"],
                score=response.get("score"),
                source_record_id=None if source_record is None else source_record.id,
                source_text=None if source_record is None else source_record.text,
                source_category=None if source_record is None else source_record.category,
                confidence_score=response.get("confidence_score"),
                rejection_reason=response.get("rejection_reason"),
                label="up",
            )
            st.success("Feedback saved.")
            st.rerun()

    with feedback_col2:
        if st.button("Incorrect / Unhelpful", use_container_width=True, key="feedback_down"):
            log_feedback(
                question=question,
                answer=response["answer"],
                score=response.get("score"),
                source_record_id=None if source_record is None else source_record.id,
                source_text=None if source_record is None else source_record.text,
                source_category=None if source_record is None else source_record.category,
                confidence_score=response.get("confidence_score"),
                rejection_reason=response.get("rejection_reason"),
                label="down",
            )
            st.success("Feedback saved.")
            st.rerun()


def handle_chat_message(store: VectorStore, message: str, min_similarity: float, min_score_gap: float) -> None:
    cleaned_message = normalize_text(message)
    if not cleaned_message:
        return

    append_chat_message("user", cleaned_message)
    intent = detect_message_intent(cleaned_message)

    if intent == "teach":
        teach_text = strip_teach_prefix(cleaned_message)
        try:
            added, record = store.add_text(teach_text, source="chat")
        except ValueError:
            append_chat_message("assistant", "I need a non-empty statement before I can store it.")
            return

        reset_answer_state()
        append_chat_message("assistant", build_teach_response(record, added))
        return

    results = search_memory(cleaned_message, store, top_k=3)
    response = build_answer_response(
        results,
        min_similarity=min_similarity,
        min_score_gap=min_score_gap,
    )
    st.session_state.last_question = cleaned_message
    st.session_state.last_response = response
    append_chat_message("assistant", response["answer"])


def render_chat_tab(store: VectorStore, min_similarity: float, min_score_gap: float) -> None:
    st.subheader("Chat")
    st.caption("Teach the app something naturally or ask it a question in the same input box.")

    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat_history"):
            st.session_state.chat_history = []
            reset_answer_state()
            st.rerun()
    with control_col2:
        st.write("")

    if not st.session_state.chat_history:
        st.info("Try messages like `Remember that my favorite music is rock.` or `What is my favorite music?`")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Teach a fact or ask a question")
    if prompt:
        handle_chat_message(store, prompt, min_similarity=min_similarity, min_score_gap=min_score_gap)
        st.rerun()

    if st.session_state.last_question and st.session_state.last_response:
        render_response(st.session_state.last_response)
        render_feedback_section(st.session_state.last_question, st.session_state.last_response)


def render_upload_tab(store: VectorStore) -> None:
    st.subheader("Document Upload")
    st.caption("Upload text, markdown, or PDF files to chunk and add them into semantic memory.")

    uploaded_files = st.file_uploader(
        "Choose files to ingest",
        type=list(SUPPORTED_UPLOAD_EXTENSIONS),
        accept_multiple_files=True,
    )
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200,
        max_value=1200,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        key="upload_chunk_size",
    )

    if st.button("Ingest Files", type="primary", disabled=not uploaded_files, key="ingest_files"):
        drafts: list[MemoryDraft] = []
        previews: list[str] = []
        failures: list[str] = []

        for uploaded_file in uploaded_files or []:
            try:
                plan = build_document_ingestion_plan(uploaded_file, chunk_size=chunk_size)
            except ValueError as exc:
                failures.append(str(exc))
                continue

            previews.append(f"{plan.name}: {len(plan.chunks)} chunks")
            drafts.extend(
                MemoryDraft(
                    text=chunk,
                    source=plan.source,
                    category="document_excerpt",
                    tags=plan.tags,
                )
                for chunk in plan.chunks
            )

        added_count, duplicate_count = store.add_many(drafts)
        reset_app_state()

        if added_count > 0:
            st.success(f"Added {added_count} new document chunks to memory.")
        if duplicate_count > 0:
            st.info(f"Skipped {duplicate_count} duplicate chunks.")
        for failure in failures:
            st.warning(failure)

        if previews:
            with st.expander("Upload Summary", expanded=True):
                for preview in previews:
                    st.write(f"- {preview}")


def render_memory_tab(store: VectorStore) -> None:
    st.subheader("Memory Browser")
    records = store.records()

    if not records:
        st.info("No memories stored yet.")
        return

    categories = sorted({record.category for record in records})
    sources = sorted({record.source for record in records})

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_categories = st.multiselect(
            "Filter by category",
            options=categories,
            default=categories,
            key="memory_filter_categories",
        )
    with filter_col2:
        selected_sources = st.multiselect(
            "Filter by source",
            options=sources,
            default=sources,
            key="memory_filter_sources",
        )

    filtered_records = [
        record
        for record in records
        if record.category in selected_categories and record.source in selected_sources
    ]

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Visible memories", len(filtered_records))
    summary_col2.metric("Categories", len({record.category for record in filtered_records}))
    summary_col3.metric("Sources", len({record.source for record in filtered_records}))

    st.dataframe(
        [
            {
                "text": record.text,
                "category": format_category_label(record.category),
                "source": format_source_label(record.source),
                "tags": ", ".join(record.tags),
                "updated_at": record.updated_at,
            }
            for record in filtered_records
        ],
        use_container_width=True,
        hide_index=True,
    )

    if not filtered_records:
        st.info("No memories match the selected filters.")
        return

    record_map = {record.id: record for record in filtered_records}
    selected_record_id = st.selectbox(
        "Select a memory to edit",
        options=list(record_map),
        format_func=lambda record_id: truncate_text(record_map[record_id].text),
        key="memory_editor_selection",
    )
    selected_record = record_map[selected_record_id]

    edited_text = st.text_area(
        "Edit memory text",
        value=selected_record.text,
        height=120,
        key="memory_editor_text",
    )
    edit_col1, edit_col2, edit_col3 = st.columns(3)
    with edit_col1:
        edited_category = st.selectbox(
            "Category",
            options=CATEGORY_OPTIONS,
            index=CATEGORY_OPTIONS.index(selected_record.category)
            if selected_record.category in CATEGORY_OPTIONS
            else 0,
            key="memory_editor_category",
        )
    with edit_col2:
        edited_source = st.text_input(
            "Source",
            value=selected_record.source,
            key="memory_editor_source",
        )
    with edit_col3:
        edited_tags = st.text_input(
            "Tags",
            value=", ".join(selected_record.tags),
            key="memory_editor_tags",
        )

    manage_col1, manage_col2 = st.columns(2)
    with manage_col1:
        if st.button("Save Changes", use_container_width=True, key="memory_editor_save"):
            try:
                status = store.update_record(
                    selected_record.id,
                    edited_text,
                    category=edited_category,
                    source=normalize_text(edited_source) or selected_record.source,
                    tags=split_tags(edited_tags),
                )
            except ValueError:
                st.warning("Please enter a non-empty memory before saving.")
            else:
                if status == "updated":
                    reset_app_state()
                    st.success("Memory updated.")
                    st.rerun()
                elif status == "duplicate":
                    st.info("That updated memory already exists.")
                else:
                    st.warning("The selected memory could not be found.")

    with manage_col2:
        if st.button("Delete Selected Memory", use_container_width=True, key="memory_editor_delete"):
            deleted = store.delete_record(selected_record.id)
            if deleted:
                reset_app_state()
                st.success("Memory deleted.")
                st.rerun()
            else:
                st.warning("The selected memory could not be found.")


def render_review_tab(store: VectorStore) -> None:
    st.subheader("Study Review")
    st.caption("Generate a quick flashcard from a stored memory and self-test before revealing the answer.")

    records = [record for record in store.records() if record.category != "document_excerpt"]
    if not records:
        st.info("Add some direct facts first, then come back here to review them.")
        return

    categories = sorted({record.category for record in records})
    selected_categories = st.multiselect(
        "Study categories",
        options=categories,
        default=categories,
        key="study_categories",
    )
    candidates = [record for record in records if record.category in selected_categories]
    if not candidates:
        st.info("No study cards match the selected categories.")
        return

    if st.button("Pick a Study Card", type="primary", key="study_pick_card"):
        chosen = random.choice(candidates)
        st.session_state.study_record_id = chosen.id
        st.session_state.study_revealed = False
        st.rerun()

    selected_record = next(
        (record for record in candidates if record.id == st.session_state.study_record_id),
        None,
    )
    if selected_record is None:
        return

    flashcard = build_flashcard(selected_record)
    st.markdown(f"**Prompt**")
    st.write(flashcard.prompt)
    st.caption(flashcard.hint)

    if st.button("Reveal Answer", use_container_width=True, key="study_reveal"):
        st.session_state.study_revealed = True
        st.rerun()

    if st.session_state.study_revealed:
        st.markdown("**Answer**")
        st.success(flashcard.answer)
        st.caption(
            f"Category: {format_category_label(selected_record.category)} | "
            f"Source: {format_source_label(selected_record.source)}"
        )


st.set_page_config(page_title="AI Learning Companion", layout="wide")
st.title("AI Learning Companion")
st.caption(
    "A local semantic memory app with structured records, cosine retrieval, chat-style interaction, and document ingestion."
)

init_session_state()
store = get_store()
records = store.records()

with st.sidebar:
    st.header("Retrieval Settings")
    min_similarity = st.slider(
        "Minimum cosine similarity",
        min_value=0.00,
        max_value=1.00,
        value=0.45,
        step=0.01,
        help="Higher values make the app stricter about when it will answer.",
    )
    min_score_gap = st.slider(
        "Minimum top-match gap",
        min_value=0.00,
        max_value=0.30,
        value=0.05,
        step=0.01,
        help="If the best and second-best matches are too close, the app will refuse to guess.",
    )

    st.header("Project Snapshot")
    st.metric("Memories", len(records))
    st.metric("Sources", len({record.source for record in records}))
    feedback_stats = get_feedback_stats()
    st.metric("Feedback entries", feedback_stats["total"])

    category_counts = Counter(record.category for record in records)
    if category_counts:
        st.caption("Memory categories")
        for category, count in category_counts.most_common():
            st.caption(f"{format_category_label(category)}: {count}")

    st.header("Controls")
    if st.button("Clear Memory", type="secondary", use_container_width=True):
        store.clear()
        reset_app_state()
        st.session_state.chat_history = []
        st.success("Memory cleared.")
        st.rerun()

    if st.button("Clear Feedback Log", type="secondary", use_container_width=True):
        clear_feedback_log()
        st.success("Feedback log cleared.")
        st.rerun()

chat_tab, upload_tab, memory_tab, review_tab = st.tabs(
    ["Chat", "Upload", "Memory", "Review"]
)

with chat_tab:
    render_chat_tab(store, min_similarity=min_similarity, min_score_gap=min_score_gap)

with upload_tab:
    render_upload_tab(store)

with memory_tab:
    render_memory_tab(store)

with review_tab:
    render_review_tab(store)

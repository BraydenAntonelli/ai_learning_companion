from __future__ import annotations

import random
from typing import Any, Dict, Optional

import streamlit as st

from feedback.logger import clear_feedback_log, get_feedback_stats, log_feedback
from llm.ollama_client import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    LocalLLMConfig,
    generate_grounded_answer,
    get_ollama_status,
)
from llm.responder import (
    build_answer_response,
    build_grounded_fallback_answer,
    build_teach_response,
    format_category_label,
)
from memory.classifier import CATEGORY_OPTIONS, detect_message_intent, strip_teach_prefix
from memory.embedder import embed_text, embed_texts, get_fallback_reason
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


@st.cache_data(ttl=5, show_spinner=False)
def get_cached_ollama_status(model: str, base_url: str, timeout_seconds: float):
    return get_ollama_status(
        LocalLLMConfig(
            enabled=True,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    )


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "chat_history": [],
        "last_question": None,
        "last_response": None,
        "notices": [],
        "last_upload_summary": [],
        "study_record_id": None,
        "study_revealed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_state_by_prefixes(prefixes: tuple[str, ...]) -> None:
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


def clear_memory_editor_state() -> None:
    clear_state_by_prefixes(("memory_editor_", "memory_filter_", "study_"))


def reset_answer_state() -> None:
    st.session_state.last_question = None
    st.session_state.last_response = None


def reset_app_state() -> None:
    reset_answer_state()
    clear_memory_editor_state()
    st.session_state.study_record_id = None
    st.session_state.study_revealed = False


def push_notice(level: str, message: str) -> None:
    st.session_state.notices.append({"level": level, "message": message})


def render_notices() -> None:
    notices = list(st.session_state.notices)
    st.session_state.notices = []
    for notice in notices:
        level = notice["level"]
        message = notice["message"]
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)


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


def build_summary_text(record_count: int, source_count: int, feedback_count: int) -> str:
    return f"{record_count} memories | {source_count} sources | {feedback_count} feedback"


def estimate_chat_spacer_height(message_count: int, has_response: bool) -> float:
    spacer_height = 26.0 - (message_count * 2.8)
    if has_response:
        spacer_height -= 5.0
    return max(0.0, spacer_height)


def inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-ink: #152321;
            --app-muted: #5d6e6a;
            --app-accent: #136f63;
            --app-accent-soft: #dcefe9;
            --app-warm: #c46d2d;
            --app-card: rgba(255, 252, 247, 0.86);
            --app-line: rgba(21, 35, 33, 0.10);
            --app-shadow: 0 20px 45px rgba(16, 33, 29, 0.08);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(19, 111, 99, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(196, 109, 45, 0.12), transparent 28%),
                linear-gradient(180deg, #f7f2e8 0%, #edf3ef 100%);
            color: var(--app-ink);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 0.45rem;
            padding-bottom: 2.2rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(250, 246, 237, 0.98), rgba(241, 247, 245, 0.98));
            border-right: 1px solid rgba(19, 111, 99, 0.10);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }

        h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.01em;
            color: var(--app-ink);
        }

        p, li, label, .stMarkdown, .stCaption {
            font-family: Aptos, "Segoe UI", "Trebuchet MS", sans-serif;
        }

        div[data-testid="stMetric"] {
            background: var(--app-card);
            border: 1px solid var(--app-line);
            border-radius: 20px;
            padding: 0.85rem 1rem;
            box-shadow: var(--app-shadow);
        }

        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.46);
            border: 1px solid rgba(21, 35, 33, 0.10);
            border-radius: 18px;
            box-shadow: 0 10px 22px rgba(16, 33, 29, 0.04);
            overflow: hidden;
        }

        div[data-testid="stExpander"] details {
            border: none;
        }

        div[data-testid="stExpander"] summary {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem;
        }

        div[data-testid="stExpander"] summary p {
            font-weight: 600;
            color: var(--app-ink);
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid rgba(21, 35, 33, 0.08);
            border-radius: 22px;
            box-shadow: 0 12px 24px rgba(16, 33, 29, 0.04);
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(21, 35, 33, 0.08);
            font-weight: 600;
            letter-spacing: 0.01em;
            box-shadow: none;
        }

        .stTextInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 16px !important;
        }

        .app-title-shell {
            margin: 0 0 0.15rem;
        }

        .app-kicker {
            font-family: Aptos, "Segoe UI", "Trebuchet MS", sans-serif;
            color: var(--app-accent);
            text-transform: uppercase;
            letter-spacing: 0.28em;
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .app-title {
            font-family: "Didot", "Bodoni MT", "Garamond", serif;
            font-size: clamp(2.2rem, 4vw, 3.25rem);
            line-height: 0.9;
            letter-spacing: -0.05em;
            color: var(--app-ink);
            font-weight: 700;
            margin: 0;
        }

        .app-title-accent {
            color: var(--app-accent);
            font-style: italic;
        }

        .chat-summary {
            padding-top: 0.3rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="app-title-shell">
            <div class="app-kicker">Aila</div>
            <div class="app-title">AI Learning <span class="app-title-accent">Aid</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response(response: Dict[str, Any]) -> None:
    source_record: Optional[MemoryRecord] = response.get("source_record")
    alternate_source: Optional[MemoryRecord] = response.get("alternate_source_record")

    if response.get("rejection_reason") == "ambiguous":
        st.info("The top two memories were too close together, so the app refused to guess.")
    elif response.get("rejection_reason") == "low_confidence" and source_record is not None:
        st.info("A related memory was found, but the similarity score was below the current cutoff.")

    with st.expander("Details"):
        if response.get("answer_mode") == "local_llm" and response.get("llm_model"):
            st.caption(f"Written with local model `{response['llm_model']}`.")
        elif response.get("answer_mode") == "template_fallback":
            st.caption("Written directly from stored memory.")
        elif response.get("llm_error"):
            st.caption("Using direct retrieval output.")
            st.caption(str(response["llm_error"]))

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Cosine similarity", format_similarity(response.get("score")))
        metric_col2.metric("Confidence score", format_confidence(response.get("confidence_score")))
        metric_col3.metric(
            "Confidence tier",
            str(response.get("confidence_label") or "n/a").title(),
        )

        if response.get("score_gap") is not None:
            st.caption(f"Gap to second-best match: {response['score_gap']:.3f}")

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

        if alternate_source is not None and response.get("rejection_reason") == "ambiguous":
            st.markdown("**Another similarly close memory**")
            st.write(alternate_source.text)
            st.caption(
                f"Category: {format_category_label(alternate_source.category)} | "
                f"Source: {format_source_label(alternate_source.source)}"
            )


def render_feedback_section(question: str, response: Dict[str, Any]) -> None:
    st.divider()
    st.caption("Was this helpful?")

    source_record: Optional[MemoryRecord] = response.get("source_record")

    feedback_col1, feedback_col2, _ = st.columns([1.15, 1.25, 7.6])
    with feedback_col1:
        if st.button("Correct", use_container_width=True, key="feedback_up"):
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
            push_notice("success", "Feedback saved.")
            st.rerun()

    with feedback_col2:
        if st.button("Incorrect", use_container_width=True, key="feedback_down"):
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
            push_notice("success", "Feedback saved.")
            st.rerun()


def handle_chat_message(
    store: VectorStore,
    message: str,
    min_similarity: float,
    min_score_gap: float,
    llm_config: LocalLLMConfig,
    llm_unavailable_message: Optional[str] = None,
) -> None:
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
    generation = generate_grounded_answer(
        cleaned_message,
        results,
        response,
        llm_config,
    )
    if generation.used and generation.answer:
        response["answer"] = generation.answer
        response["answer_mode"] = "local_llm"
        response["llm_model"] = generation.model
    elif llm_config.enabled and response.get("found") and response.get("source_record") is not None:
        response["answer"] = build_grounded_fallback_answer(
            cleaned_message,
            response["source_record"],
        )
        response["answer_mode"] = "template_fallback"
        response["llm_model"] = None
    else:
        response["answer_mode"] = "retrieval"
        response["llm_model"] = None
    response["llm_error"] = generation.error or llm_unavailable_message
    response["llm_usage"] = generation.usage

    st.session_state.last_question = cleaned_message
    st.session_state.last_response = response
    append_chat_message("assistant", response["answer"])


def render_chat_tab(
    store: VectorStore,
    min_similarity: float,
    min_score_gap: float,
    llm_config: LocalLLMConfig,
    summary_text: str,
    llm_unavailable_message: Optional[str] = None,
) -> None:
    has_response = bool(st.session_state.last_question and st.session_state.last_response)
    spacer_height = estimate_chat_spacer_height(
        len(st.session_state.chat_history),
        has_response,
    )
    if spacer_height > 0:
        st.markdown(
            f"<div style='height: {spacer_height}rem;'></div>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.last_question and st.session_state.last_response:
        render_response(st.session_state.last_response)

    with st.form("chat_input_form", clear_on_submit=True):
        composer_col, send_col = st.columns([9.5, 1.1])
        with composer_col:
            prompt = st.text_input(
                "Teach Aila a fact or ask a question",
                placeholder="Teach Aila a fact or ask a question",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and prompt:
        handle_chat_message(
            store,
            prompt,
            min_similarity=min_similarity,
            min_score_gap=min_score_gap,
            llm_config=llm_config,
            llm_unavailable_message=llm_unavailable_message,
        )
        st.rerun()

    if st.session_state.last_question and st.session_state.last_response:
        render_feedback_section(st.session_state.last_question, st.session_state.last_response)

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
    footer_left, footer_right = st.columns([8.2, 1.35])
    with footer_left:
        st.markdown(
            f"<div class='chat-summary'>{summary_text}</div>",
            unsafe_allow_html=True,
        )
    with footer_right:
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat_history"):
            st.session_state.chat_history = []
            reset_answer_state()
            st.rerun()


def render_upload_tab(store: VectorStore) -> None:
    if st.session_state.last_upload_summary:
        with st.expander("Last Upload Summary", expanded=True):
            for preview in st.session_state.last_upload_summary:
                st.write(f"- {preview}")

    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
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
        st.session_state.last_upload_summary = previews

        if added_count > 0:
            push_notice("success", f"Added {added_count} new document chunks to memory.")
        if duplicate_count > 0:
            push_notice("info", f"Skipped {duplicate_count} duplicate chunks.")
        for failure in failures:
            push_notice("warning", failure)
        if added_count == 0 and duplicate_count == 0 and not failures:
            push_notice("info", "No new chunks were added from the selected files.")
        st.rerun()


def render_memory_tab(store: VectorStore) -> None:
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
                "memory": truncate_text(record.text, max_chars=96),
                "category": format_category_label(record.category),
                "source": format_source_label(record.source),
                "tags": ", ".join(record.tags),
                "updated": record.updated_at,
            }
            for record in filtered_records
        ],
        use_container_width=True,
        hide_index=True,
        height=320,
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
                push_notice("warning", "Please enter a non-empty memory before saving.")
                st.rerun()
            else:
                if status == "updated":
                    reset_app_state()
                    push_notice("success", "Memory updated.")
                    st.rerun()
                elif status == "duplicate":
                    push_notice("info", "That updated memory already exists.")
                    st.rerun()
                else:
                    push_notice("warning", "The selected memory could not be found.")
                    st.rerun()

    with manage_col2:
        if st.button("Delete Selected Memory", use_container_width=True, key="memory_editor_delete"):
            deleted = store.delete_record(selected_record.id)
            if deleted:
                reset_app_state()
                push_notice("success", "Memory deleted.")
                st.rerun()
            else:
                push_notice("warning", "The selected memory could not be found.")
                st.rerun()


def render_review_tab(store: VectorStore) -> None:
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
        if st.session_state.study_record_id is not None:
            st.session_state.study_record_id = None
            st.session_state.study_revealed = False
            st.info("Pick a study card from the currently visible categories.")
        return

    flashcard = build_flashcard(selected_record)
    st.write(flashcard.prompt)
    st.caption(f"Hint: {flashcard.hint}")

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

st.set_page_config(
    page_title="AI Learning Aid",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_ui_styles()

init_session_state()
render_notices()
store = get_store()
records = store.records()
feedback_stats = get_feedback_stats()
source_count = len({record.source for record in records})
summary_text = build_summary_text(len(records), source_count, feedback_stats["total"])

render_hero()
current_answer_mode = st.session_state.get("answer_mode", "retrieval")

with st.expander("Settings"):
    st.caption("Retrieval")
    retrieval_col1, retrieval_col2 = st.columns(2)
    with retrieval_col1:
        min_similarity = st.slider(
            "Minimum cosine similarity",
            min_value=0.00,
            max_value=1.00,
            value=0.45,
            step=0.01,
            help="Higher values make the app stricter about when it will answer.",
        )
    with retrieval_col2:
        min_score_gap = st.slider(
            "Minimum top-match gap",
            min_value=0.00,
            max_value=0.30,
            value=0.05,
            step=0.01,
            help="If the best and second-best matches are too close, the app will refuse to guess.",
        )

    llm_model = DEFAULT_OLLAMA_MODEL
    llm_base_url = DEFAULT_OLLAMA_BASE_URL
    if current_answer_mode == "local_llm":
        st.caption("Local LLM")
        llm_col1, llm_col2 = st.columns(2)
        with llm_col1:
            llm_model = st.text_input(
                "Local model",
                value=DEFAULT_OLLAMA_MODEL,
                help="Example: `llama3.2:3b`, `gemma3`, or another Ollama model you already pulled.",
                key="llm_model",
            )
        with llm_col2:
            llm_base_url = st.text_input(
                "Local LLM URL",
                value=DEFAULT_OLLAMA_BASE_URL,
                help="Default Ollama URL on your own machine.",
                key="llm_base_url",
            )

    fallback_reason = get_fallback_reason()
    if fallback_reason:
        st.warning(fallback_reason)

    st.caption("Maintenance")
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        if st.button("Rebuild Embeddings", type="secondary", use_container_width=True):
            rebuilt_count = store.rebuild_index()
            reset_app_state()
            push_notice("success", f"Rebuilt embeddings for {rebuilt_count} stored memories.")
            st.rerun()
    with control_col2:
        if st.button("Clear Feedback Log", type="secondary", use_container_width=True):
            clear_feedback_log()
            push_notice("success", "Feedback log cleared.")
            st.rerun()
    with control_col3:
        if st.button("Clear Memory", type="secondary", use_container_width=True):
            store.clear()
            reset_app_state()
            st.session_state.chat_history = []
            st.session_state.last_upload_summary = []
            push_notice("success", "Memory cleared.")
            st.rerun()

view_col, mode_col = st.columns([1.65, 0.95])
with view_col:
    current_view = st.radio(
        "View",
        options=("Chat", "Upload", "Memory", "Review"),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="current_view",
    )
with mode_col:
    answer_mode = st.radio(
        "Mode",
        options=("retrieval", "local_llm"),
        format_func=lambda value: (
            "Direct answer" if value == "retrieval" else "Local LLM"
        ),
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="answer_mode",
    )

llm_config = LocalLLMConfig(enabled=False)
llm_unavailable_message: Optional[str] = None
llm_status = None
if answer_mode == "local_llm":
    llm_config = LocalLLMConfig(
        enabled=True,
        model=llm_model,
        base_url=llm_base_url,
    )
    llm_status = get_cached_ollama_status(
        llm_config.normalized_model,
        llm_config.normalized_base_url,
        llm_config.timeout_seconds,
    )
    if llm_status.available:
        status_bits = [f"Using {llm_config.normalized_model}"]
        if llm_status.version:
            status_bits.append(f"Ollama {llm_status.version}")
        st.caption(" | ".join(status_bits))
    else:
        llm_unavailable_message = llm_status.message
        llm_config = LocalLLMConfig(
            enabled=False,
            model=llm_model,
            base_url=llm_base_url,
        )
        st.warning(llm_status.message)

if current_view == "Chat":
    render_chat_tab(
        store,
        min_similarity=min_similarity,
        min_score_gap=min_score_gap,
        llm_config=llm_config,
        summary_text=summary_text,
        llm_unavailable_message=llm_unavailable_message,
    )
elif current_view == "Upload":
    render_upload_tab(store)
elif current_view == "Memory":
    render_memory_tab(store)
else:
    render_review_tab(store)

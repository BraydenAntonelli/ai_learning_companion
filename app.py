import streamlit as st
from memory.embedder import embed_text
from memory.vector_store import VectorStore
from retriever.semantic_search import search_memory

# Set page title
st.set_page_config(page_title="AI Learning Companion")

st.title("🧠 AI Learning Companion")

def get_store():
    return VectorStore(
        dim=384,
        index_path="data/memory.faiss",
        metadata_path="data/memory.json"
    )

# Section 1: Teach
st.header("📚 Teach Me Something")

with st.form(key="teach_form"):
    fact_input = st.text_area("Enter a fact to remember:")
    teach_submit = st.form_submit_button("Teach")
    if teach_submit and fact_input.strip():
        store = get_store()
        store.add(embed_text(fact_input), fact_input)
        st.success("✅ Learned: I’ll remember that.")

# Section 2: Ask
st.header("❓ Ask a Question")

MAX_DISTANCE = 1.0  # threshold for confident answer

question = st.text_input("What would you like to know?")
if question.strip():
    store = get_store()
    results = search_memory(question, store)

    if results:
        best_fact, dist = results[0]
        if dist <= MAX_DISTANCE:
            st.subheader("🧠 Best Answer")
            st.markdown(f"{best_fact}")
            st.caption(f"(semantic distance: {round(dist, 4)})")
        else:
            st.warning("🤔 I'm not confident enough to answer that. Try teaching me first!")
    else:
        st.warning("🤔 I don’t know that yet. Try teaching me first!")

# AI Learning Companion

AI Learning Companion is a local Streamlit app for building and querying semantic memory. You can teach it facts directly in chat, upload notes or PDFs for ingestion, browse and edit stored memories, and review what it knows through a lightweight study mode.

The project demonstrates a fuller AI application pipeline than the original MVP:

- structured memory records instead of raw strings
- normalized embeddings with cosine-similarity retrieval
- confidence-aware answer selection
- source-grounded responses with metadata
- document upload and chunk ingestion
- chat-style teach-or-ask interaction
- memory browsing, editing, and deletion
- review cards generated from stored knowledge

## What The App Does

1. Accepts natural-language input in a unified chat interface.
   Statements are stored as memory and questions are treated as retrieval queries.
2. Classifies stored memories automatically.
   New memories are labeled as personal context, academic concepts, factual statements, temporary notes, question-like inputs, or document excerpts.
3. Converts memories into local embeddings.
   The app uses `sentence-transformers` locally, so no paid API key is required.
4. Stores memory persistently in FAISS plus structured JSON metadata.
   Each record keeps text, category, source, tags, and timestamps.
5. Uses cosine similarity for semantic retrieval.
   Query and memory embeddings are normalized before search.
6. Rejects weak or ambiguous matches.
   The app uses both a minimum similarity threshold and a top-match gap threshold.
7. Ingests uploaded documents.
   Text, markdown, and PDF files can be chunked and added into memory.
8. Supports lightweight study review.
   Stored memories can be surfaced as simple flashcard-style prompts.

## Main Features

- chat-style memory teaching and question answering
- automatic memory classification
- structured memory metadata with timestamps, source labels, and tags
- cosine-similarity retrieval with configurable thresholds
- document upload for `.txt`, `.md`, and `.pdf`
- memory browser with filters, editing, and deletion
- feedback logging for retrieval quality
- study review cards generated from saved memories

## Example Workflow

1. Chat: `Remember that my favorite music is rock.`
2. Chat: `What is my favorite music?`
3. The app retrieves the strongest memory match and decides whether it is confident enough to answer.
4. Upload a study document and let the app chunk it into memory.
5. Open the Memory tab to inspect records by category or source.
6. Open the Review tab to test yourself on stored memories.

## Tech Stack

- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`
- `pypdf`

## Project Structure

```text
ai_learning_companion/
|-- app.py
|-- feedback/
|   |-- __init__.py
|   `-- logger.py
|-- llm/
|   |-- __init__.py
|   `-- responder.py
|-- memory/
|   |-- __init__.py
|   |-- classifier.py
|   |-- embedder.py
|   |-- models.py
|   `-- vector_store.py
|-- retriever/
|   |-- __init__.py
|   `-- semantic_search.py
|-- test/
|   |-- support.py
|   |-- test_classifier.py
|   |-- test_document_utils.py
|   |-- test_embed.py
|   |-- test_feedback_logger.py
|   |-- test_responder.py
|   |-- test_semantic_search.py
|   |-- test_study_utils.py
|   |-- test_teach_and_ask.py
|   |-- test_vector_store.py
|   `-- test_vector_store_2.py
|-- utils/
|   |-- __init__.py
|   |-- document_utils.py
|   |-- paths.py
|   |-- study_utils.py
|   `-- text_utils.py
|-- data/
|   `-- memory.json
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
python -m streamlit run app.py
```

### Tabs In The App

- `Chat`: teach or ask with one unified input
- `Upload`: ingest text, markdown, or PDF documents
- `Memory`: browse, filter, edit, and delete records
- `Review`: generate simple study cards from stored memories

## Run The Tests

```bash
python -m unittest discover -s test
```

The test suite covers:

- embedding normalization behavior
- memory classification
- structured vector-store persistence and migration
- semantic search flow
- answer-response logic
- feedback logging
- document-ingestion planning
- flashcard generation

## Data Model

Each memory record stores:

- text
- category
- source
- tags
- created timestamp
- updated timestamp

Embeddings live in `data/memory.faiss`, while structured metadata lives in `data/memory.json`.

## Optimization Highlights

- batch embedding support for document ingestion and index rebuilds
- lazy model loading to avoid expensive imports before first use
- normalized vector handling for stable cosine similarity
- metadata migration from older raw-string memory files
- word-aware chunking for uploaded documents

## Troubleshooting

- If `streamlit` is not recognized, run the app with `python -m streamlit run app.py`.
- If PDF uploads fail, make sure `pypdf` is installed from `requirements.txt`.
- If retrieval is too strict, lower the minimum similarity or top-match gap in the sidebar.
- If `python` on Windows points to the Microsoft Store shim, activate a real virtual environment first and use that interpreter.

## License

This project is currently unlicensed. Add a `LICENSE` file before publishing if you want to define reuse terms clearly.

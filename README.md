# AI Learning Aid

AI Learning Aid is a local semantic-memory and grounded LLM app built around retrieval, persistence, and grounded answering. The assistant inside the app is called `Aila`, based on the initials in the project name.

You can:

- teach Aila facts, preferences, notes, and short concepts in chat
- ask questions in natural language and retrieve the closest stored memory
- upload `.txt`, `.md`, and `.pdf` files for ingestion
- browse, edit, and delete stored memories
- rate answers as `Correct` or `Incorrect`
- optionally use a free local Ollama model to turn retrieved memory into a more natural answer

## What This Project Is

At its core, this is a local retrieval system with structured memory.

The current app:

- uses `sentence-transformers` to embed text locally
- stores vector search data in `FAISS`
- stores structured memory and feedback data in `SQLite`
- uses cosine similarity and confidence thresholds to decide whether to answer
- can optionally pass grounded retrieval results into a local LLM through `Ollama`

The local LLM mode is an answer layer on top of retrieval. Aila still depends on stored memory and retrieved context, rather than answering from an unrestricted model prompt alone.

## Main Features

- unified chat flow for teaching and asking
- automatic memory classification
- persistent structured memory records
- cosine-similarity retrieval with configurable thresholds
- ambiguity and low-confidence rejection
- optional grounded local LLM answers through Ollama
- deterministic fallback answers when the local model is unavailable or weak
- document upload and chunk ingestion
- smarter upload splitting for lines, bullets, numbered lists, and long paragraphs
- memory filtering, editing, and deletion
- feedback logging in SQLite
- lightweight study/review cards

## How It Works

1. You type a statement or a question.
2. Aila decides whether the message looks like something to remember or something to answer.
3. Stored text is embedded with a local `sentence-transformers` model.
4. Embeddings are indexed in `FAISS` for semantic search.
5. Memory records and feedback are stored in `SQLite`.
6. A question is embedded and compared against stored memory.
7. The app either:
   - returns the retrieved memory directly, or
   - sends grounded context into a local Ollama model for a more natural response
8. If the best match is too weak or too ambiguous, Aila refuses to guess.

## Current App Layout

- `Chat`
  Teach Aila a fact or ask a question in one place.
- `Upload`
  Ingest `.txt`, `.md`, and `.pdf` files into memory.
- `Memory`
  Browse, filter, edit, and delete saved records.
- `Review`
  Surface saved memories as simple study prompts.

The main screen also includes:

- a top view switch for `Chat`, `Upload`, `Memory`, and `Review`
- a top answer-mode switch for `Direct answer` and `Local LLM`
- a `Settings` expander for retrieval thresholds, local LLM settings, and maintenance actions

## Example Flow

1. Type: `My favorite food is pizza.`
2. Aila stores it as a memory record.
3. Ask: `What is my favorite food?`
4. The app embeds the question and retrieves the closest memory.
5. If the match is strong enough, Aila answers.
6. You can rate the answer with `Correct` or `Incorrect`.

## Storage

The app now uses a split storage model:

- [data/memory.sqlite3](data/memory.sqlite3)
  Structured memory records and feedback records
- [data/memory.faiss](data/memory.faiss)
  Vector index used for semantic retrieval

The live app uses SQLite + FAISS.

## Data Model

Each memory record stores:

- `id`
- `text`
- `category`
- `source`
- `tags`
- `created_at`
- `updated_at`

Feedback entries are stored in their own SQLite table and include:

- `timestamp`
- `question`
- `answer`
- `label`
- `score`
- `source_record_id`
- `source_text`
- `source_category`
- `confidence_score`
- `rejection_reason`

## Tech Stack

- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`
- `pypdf`
- `sqlite3` from the Python standard library
- optional local `Ollama` runtime

## Setup

Python `3.10+` is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
python -m streamlit run app.py
```

## Optional Free Local LLM Mode

If you want grounded generated answers without paying for an API, install Ollama and pull a local model.

### Install Ollama

Windows PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

macOS:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull a model:

```bash
ollama pull llama3.2:3b
```

Then switch the app from `Direct answer` to `Local LLM`.

`llama3.2:3b` is a good default here because it is small enough to run locally while still giving clean grounded rewrites.

### Local LLM Behavior

When `Local LLM` is enabled, the app:

1. retrieves the strongest memory matches first
2. checks whether retrieval is strong enough to answer
3. sends grounded memory context into Ollama
4. returns a more natural response if the model behaves well
5. falls back to deterministic memory-based phrasing if Ollama is unavailable or the response is weak

The first generated response can take longer while Ollama loads the model into memory.

## Run The Tests

```bash
python -m unittest discover -s test
```

The test suite covers:

- embedding behavior
- memory classification
- vector-store persistence and rebuild behavior
- semantic search
- answer-response logic
- local Ollama helpers
- feedback logging
- document ingestion
- study-card generation

## Project Structure

```text
ai_learning_companion/
|-- app.py
|-- feedback/
|   |-- __init__.py
|   `-- logger.py
|-- llm/
|   |-- __init__.py
|   |-- ollama_client.py
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
|   |-- test_ollama_client.py
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
|   |-- memory.faiss
|   `-- memory.sqlite3
|-- requirements.txt
`-- README.md
```

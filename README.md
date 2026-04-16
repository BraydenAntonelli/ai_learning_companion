# AI Learning Companion

AI Learning Companion is a local Streamlit app that demonstrates the core mechanics of memory-based AI retrieval. The user teaches the system short facts in plain language, the app converts those facts into embeddings, stores them persistently, and later retrieves the closest match when the user asks a related question.

This project is intentionally positioned as a strong GitHub MVP:

- easy to understand quickly
- realistic enough to show an end-to-end AI workflow
- local-first, so it runs without a paid API key
- modular enough to extend into richer retrieval systems later

## MVP

The MVP goals for this project are:

1. Teach the system information through user input.
   Users can type in facts, preferences, definitions, or notes and store them as memory.
2. Convert user input into embeddings.
   Each fact is transformed into a semantic vector using a local embedding model so the app compares meaning instead of exact wording.
3. Store memory persistently.
   Embeddings are stored in FAISS, while the original fact text is stored as metadata.
4. Let the user ask natural-language questions.
   Questions can be phrased differently from the original stored fact.
5. Use semantic search to find the closest answer.
   The app embeds the question, compares it against stored memory, and retrieves the best-matching fact.
6. Return a single best answer.
   The app behaves like a simple question-answering assistant rather than showing a long ranked list.
7. Reject weak matches.
   If the best match is too far away semantically, the app returns a fallback response instead of guessing.
8. Provide an interactive Streamlit interface.
   The UI includes a teaching section, a question section, and a displayed answer or fallback response.
9. Preserve memory between runs.
   Facts remain available in later sessions unless the user clears memory.
10. Include basic feedback logging.
    The app logs whether a response was helpful or not for later inspection.

## What The MVP Demonstrates

This project is meant to show working knowledge of:

- embeddings
- semantic similarity
- vector databases
- retrieval systems
- confidence thresholds
- interactive AI application design

That makes it a strong learning project and a solid portfolio project.

## Example Flow

Teach the app:

```text
My favorite music is rock.
The Battle of Hastings occurred in 1066.
Photosynthesis converts sunlight into energy.
```

Ask the app:

```text
What is my favorite music?
What happened in 1066?
How do plants turn sunlight into energy?
```

The app embeds the question, searches memory semantically, and returns the closest stored fact if the match is confident enough.

## Current Features

- Streamlit UI for teaching and asking
- local embeddings with `sentence-transformers`
- FAISS-backed persistent memory
- duplicate-prevention for repeated facts
- configurable confidence cutoff
- configurable ambiguity buffer based on the gap between the top two matches
- fallback response for weak matches
- ambiguity rejection when multiple memories are too close together
- source-grounded answer display with the originating memory
- heuristic confidence score and confidence tier display
- persistent feedback logging
- stored-facts viewer
- edit and delete controls for stored memory
- clear-memory and clear-feedback controls

## Tech Stack

- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`

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
|   |-- embedder.py
|   `-- vector_store.py
|-- retriever/
|   |-- __init__.py
|   `-- semantic_search.py
|-- test/
|   |-- support.py
|   |-- test_embed.py
|   |-- test_feedback_logger.py
|   |-- test_responder.py
|   |-- test_semantic_search.py
|   |-- test_teach_and_ask.py
|   |-- test_vector_store.py
|   `-- test_vector_store_2.py
|-- utils/
|   |-- __init__.py
|   |-- paths.py
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

Once the app opens:

1. Teach the system a fact in the left column.
2. Ask a question in the right column.
3. Adjust the confidence cutoff and ambiguity buffer in the sidebar if needed.
4. Review the answer, source memory, and confidence details.
5. Leave feedback on whether the answer was helpful.
6. Edit or delete stored facts from the memory-management section.

## Run The Tests

```bash
python -m unittest discover -s test
```

The automated tests cover:

- embedding input handling
- vector-store add and search behavior
- vector-store edit and delete behavior
- persistence and recovery behavior
- semantic search flow
- teach-and-ask response logic
- feedback logging
- answer-response formatting and ambiguity handling

## Data Files

- `data/memory.json` stores the original fact text
- `data/memory.faiss` stores the vector index
- `data/feedback_log.jsonl` stores answer feedback plus source and confidence metadata

The first real embedding request downloads the local transformer model, so the first run will be slower than later runs.

## Why This Works Well On GitHub

- It shows real AI application mechanics without hiding everything inside a notebook.
- It demonstrates modular thinking: UI, retrieval, storage, and feedback are separated cleanly.
- It is easy for other people to clone, run, inspect, and extend.
- It stays honest about scope: this is a retrieval MVP, not an overclaimed general assistant.

## Implemented Beyond The MVP

These stretch-goal style upgrades are already included:

- confidence-aware answer selection using both top-match distance and the gap to the second-best result
- source-grounded answers that show which stored memory the response came from
- heuristic confidence scoring to make retrieval behavior easier to inspect
- memory-management tools for editing and deleting stored facts

## Next Stretch Goals

These are the strongest next-step improvements if you want the project to look more advanced later.

1. Retrieval-augmented generation.
   Retrieve the best memory, then pass it into an LLM to generate a more polished answer.
2. Structured memory entries.
   Store richer metadata such as category, timestamp, source, confidence, and tags.
3. Memory type classification.
   Classify whether an input is a preference, factual statement, concept, note, or mistaken question.
4. Cosine-similarity retrieval.
   Normalize embeddings and switch from raw L2 distance to cosine similarity for easier scoring and debugging.
5. SQLite-backed metadata storage.
   Keep FAISS for vectors but move metadata to SQLite for more scalable updates and queries.
6. Intent-aware chat mode.
   Replace the separate Teach and Ask sections with a unified chat interface.
7. Document upload.
   Support PDFs, notes, or other files by extracting, chunking, embedding, and storing their text.
8. Web search fallback.
   If memory retrieval fails, optionally retrieve information from the web as a secondary source.
9. Quiz mode.
   Generate questions from stored facts to test the user.
10. Flashcard generation.
    Convert stored knowledge into study cards automatically.
11. Daily review or spaced repetition.
    Surface facts again over time for reinforcement.
12. Evaluation pipeline.
    Add retrieval accuracy, top-1 accuracy, rejection quality, and threshold-tuning metrics.

## Troubleshooting

- If `streamlit` is not recognized, run the app with `python -m streamlit run app.py`.
- If retrieval feels too strict, increase the cutoff slightly in the sidebar.
- If you change the retrieval setup and want a clean slate, clear memory from the UI.
- If `python` on Windows points to the Microsoft Store shim, activate a real virtual environment first and use that interpreter.

## License

This project is currently unlicensed. Add a `LICENSE` file before publishing if you want to define reuse terms clearly.

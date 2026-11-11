# Simple RAG: personal course exercise

This branch preserves the working files from the local `project/rag-gist` branch,
including its original commit `24c449a` and the previously uncommitted `SimpleRAG` work.

## Setup

Use Python 3.11 or later and uv. Create a local `.env` from `.env.example` and
configure OpenAI and Pinecone credentials. The Pinecone index must have a
`chunk_text` string field configured with `embed` in its document schema.
Both ingestion and retrieval use namespace `blog` and Pinecone Documents API.

```bash
uv sync --locked
uv run python SimpleRAG/ingestion.py
uv run python SimpleRAG/main.py
```

Ingestion writes the bundled article to Pinecone and may replace documents with
the same IDs. Retrieval uses the model configured on `chunk_text` for query
embeddings; answer generation uses the model specified in `SimpleRAG/main.py`.
These commands call external APIs and can incur usage charges.

## Migration adjustments

- Removed API-key output from ingestion.
- Resolved the article path relative to the ingestion script.
- Made ingestion failures produce a non-successful process exit.
- Fixed the direct retrieval variant to pass the search response to `format_docs`.
- Removed an unused `langchain_classic` import that was absent from the declared dependencies.
- Kept local `.env`, virtual environments, and IDE files out of Git.

## Branch organization

- `main`: course baseline.
- `react`: the four local ReAct/tool-calling examples; see `STUDY.md` there.
- `simple-rag`: this exercise.
- `projects/hello-world`, `projects/pydantic`, `projects/search`: preserved local exercises.

Create `advanced-rag` when starting the next exercise. Separate Git worktrees
can keep exercises open in different directories without switching branches.

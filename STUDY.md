# Personal course work: ReAct and tool calling

This branch preserves the local `main` work from commit `9926406`.
The four numbered Python scripts compare LangChain tool calling, native OpenAI
function calling, tool descriptions, and a manual text-based ReAct loop.

## Run

Use Python 3.13 or later, install uv, and create a local `.env` from `.env.example`.

```bash
uv sync --locked
uv run python 1_agent_loop_langchain_tool_calling.py
uv run python 2_agent_loop_raw_function_calling.py
uv run python 3_raw_react_prompt.py
uv run python 4_raw_react_prompt.py
```

The examples retain their original model names. Model availability and API access
must be checked before running; model calls can incur usage charges.

## Other study branches

- `simple-rag`: Pinecone Documents API ingestion and retrieval, with LCEL and direct variants.
- `projects/hello-world`: the original local summarization exercise.
- `projects/pydantic`: the original local structured-output exercise.
- `projects/search`: the original local search-agent exercise.
- `main`: the fork's course baseline.

`advanced-rag` is reserved as a name for a future exercise; no implementation is claimed.
The migration removes tracked IDE settings from this branch's current tree.
The original learning code is preserved.

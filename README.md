# local-code-context

Fully local MCP server for code-context retrieval. Point it at a codebase directory and it handles scanning, splitting, embedding, and hybrid search automatically.

Uses [Qwen3-Embedding-0.6B](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-mxfp8) via Apple MLX for embeddings, with hybrid search combining sqlite-vec (dense KNN) and FTS5 (sparse full-text) fused through Reciprocal Rank Fusion.

Everything runs on-device -- no API keys, no network calls, no data leaves your machine.

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone git@github.com:tteles-cf/code-context-mcp-mlx.git
cd code-context-mcp-mlx
uv sync
```

The embedding model (~600MB) is downloaded automatically on first use from Hugging Face.

## Usage

### As an MCP server

Add to your MCP client configuration (e.g. Claude Desktop, OpenCode):

```json
{
  "mcpServers": {
    "local-code-context": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/local-code-context", "mcp", "run", "src/local_code_context/server.py"]
    }
  }
}
```

The server communicates over stdio and exposes the tools listed below.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CODE_CONTEXT_DB` | `code-context.db` | Path to the SQLite database file |

## MCP Tools

### `index_codebase(path, force=False)`

Index a codebase directory for code search. Scans for source files, splits them into chunks, embeds them, and stores them for hybrid search.

- Runs **in the background** -- returns immediately. Use `get_indexing_status` to check progress.
- **Incremental by default** -- only re-indexes files that changed since the last run. Detects added, modified, and removed files via content hashing.
- Use `force=True` to drop the existing index and re-index everything from scratch.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `string` | yes | -- | Absolute path to the codebase directory |
| `force` | `boolean` | no | `false` | Force full re-index from scratch |

### `search_code(path, query, limit=10)`

Search for code in an indexed codebase using natural language.

- Uses hybrid search (vector similarity + full-text matching) with RRF fusion.
- Falls back to dense-only search if the collection is not hybrid.
- Returns code snippets with file path, line range, language, and score.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `string` | yes | -- | Absolute path to the codebase directory |
| `query` | `string` | yes | -- | Natural language search query |
| `limit` | `number` | no | `10` (max 50) | Maximum number of results |

### `clear_index(path)`

Remove the index for a codebase directory. Drops all indexed chunks and tracking state.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `string` | yes | -- | Absolute path to the codebase directory |

### `get_indexing_status(path)`

Check the indexing status of a codebase directory.

Returns one of:
- **not_found** -- not indexed yet
- **indexing** -- in progress, with percentage and file/chunk counts
- **indexed** -- complete, with stats and duration
- **failed** -- failed, with error message

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | `string` | yes | -- | Absolute path to the codebase directory |

## Workflow

```
1. "Index my project"     ->  index_codebase(path="/my/project")
2. "Check progress"       ->  get_indexing_status(path="/my/project")
3. "Find auth handlers"   ->  search_code(path="/my/project", query="authentication handlers")
4. "Re-index after edits" ->  index_codebase(path="/my/project")        # incremental
5. "Start fresh"          ->  index_codebase(path="/my/project", force=True)
6. "Remove the index"     ->  clear_index(path="/my/project")
```

## Architecture

```
src/local_code_context/
  server.py       MCP server -- 4 tools, background indexing orchestration
  splitter.py     File scanner and line-based code splitter
  snapshot.py     Indexing state persistence (~/.local-code-context/snapshots.json)
  embeddings.py   Qwen3-Embedding-0.6B via mlx-embeddings, thread-safe lazy loading
  search.py       Dense and hybrid search orchestration
  rrf.py          Reciprocal Rank Fusion algorithm
  store.py        SQLite + sqlite-vec + FTS5 storage layer
```

### How it works

1. **Indexing** -- `index_codebase` scans the directory for source files, splits them into ~80-line overlapping chunks, embeds each chunk into a 1024-d vector using Qwen3-Embedding-0.6B (mxfp8 via MLX), and stores vectors in sqlite-vec with text in FTS5. Runs in a background thread. Incremental re-indexing uses MD5 content hashes to detect changes.

2. **Search** -- `search_code` embeds the query, runs both dense KNN (sqlite-vec) and sparse FTS5 matching, then fuses results using Reciprocal Rank Fusion (k=60).

3. **State** -- Indexing progress and file hashes are persisted to `~/.local-code-context/snapshots.json` so state survives server restarts. Collection names are auto-derived from `md5(absolute_path)` -- users never need to manage collections directly.

4. **Startup** -- The embedding model is warmed in a background thread so the server accepts MCP connections immediately (~1s load, ~1.3GB RSS on Apple Silicon).

### Supported file types

`.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`, `.cpp`, `.c`, `.h`, `.hpp`, `.cs`, `.go`, `.rs`, `.php`, `.rb`, `.swift`, `.kt`, `.scala`, `.m`, `.mm`, `.md`, `.markdown`, `.toml`, `.yaml`, `.yml`, `.json`, `.sh`, `.bash`, `.zsh`

### Default ignored directories

`.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `dist`, `build`, `.next`, `.nuxt`, `target`, `.egg-info`

## Tests

```bash
uv run pytest -v
```

67 tests covering the splitter, snapshot store, embeddings, RRF fusion, and SQLite store.

## License

MIT

# Chess LMM Framework

## Overview

Framework for playing chess against Claude (or other LMMs) using a chess MCP server interface. The framework includes a mock server (python-chess), recording/observability, human CLI player, and LMM agent.

## Architecture

- `src/chess_lmm/types.py` — TypedDict response types matching SPEC.md Section 8
- `src/chess_lmm/mcp_interface.py` — `ChessSessionClient` and `ChessServerFactory` Protocols
- `src/chess_lmm/mock_server.py` — In-memory mock using python-chess
- `src/chess_lmm/recording.py` — JSON-lines recorder + RecordingClient decorator
- `src/chess_lmm/llm_agent.py` — Claude agentic loop
- `src/chess_lmm/human_player.py` — CLI human player
- `src/chess_lmm/orchestrator.py` — Game lifecycle + CLI args
- `src/chess_lmm/__main__.py` — Entry point (`python -m chess_lmm`)

## Development

```bash
uv sync --extra dev          # Install all dependencies
uv run pytest --cov          # Run tests with coverage
uv run mypy src/             # Type check
uv run ruff check src/ tests/  # Lint
uv run ruff format src/ tests/ # Format
```

## Spec Reference

The MCP interface follows `code-samples/chess/SPEC.md` Section 8. The spec defines tool parameters, response schemas, error codes, and state machine transitions.

## Testing

Test-first development. All new code must have tests. Run the full suite before submitting PRs.

## Key Design Decisions

- **Protocol over ABC**: `ChessSessionClient` uses `Protocol` (structural typing) so mock and real implementations don't need to inherit from a base class.
- **McpError exception**: All MCP tool errors are raised as `McpError` with a `code` field matching the spec error codes.
- **Per-session clients**: Each player gets their own `ChessSessionClient` instance. The server factory creates them.

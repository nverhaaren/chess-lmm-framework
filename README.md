# Chess LMM Framework

Play chess against Claude (or other LLMs) through a chess MCP server interface.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Play as white against Claude
python -m chess_lmm --color white

# Play with random color assignment
python -m chess_lmm --color random

# Custom starting position
python -m chess_lmm --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Specify model and log directory
python -m chess_lmm --model claude-sonnet-4-5-20250514 --log-dir ./game-logs
```

## Architecture

The framework separates concerns into layered components:

- **Orchestrator** — CLI entry point, game lifecycle management
- **Players** — Human (stdin/stdout) and LMM (Claude API) implementations
- **Recording** — JSON-lines logging of all MCP calls and LMM interactions
- **MCP Interface** — Protocol abstraction (`ChessSessionClient`) that both mock and real servers implement
- **Mock Server** — In-memory chess server using `python-chess` for rules

## Development

```bash
uv sync --extra dev            # Install dependencies
uv run pytest --cov            # Run tests with coverage
uv run mypy src/               # Type check
uv run ruff check src/ tests/  # Lint
```

## Spec

The MCP tool interface follows the chess engine specification in `code-samples/chess/SPEC.md` Section 8.

## Status

**Initial spike** — mock server, recording, human CLI, and LMM agent are being built incrementally.

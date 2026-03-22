# Chess LMM Framework

Play chess against Claude (or other LLMs) through a chess MCP server interface.

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** for package management
- **Anthropic API key** — set the `ANTHROPIC_API_KEY` environment variable:

  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```

  The framework uses the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python),
  which reads this variable automatically.

## Installation

```bash
git clone https://github.com/nverhaaren/chess-lmm-framework.git
cd chess-lmm-framework
uv sync
```

For development (tests, linting, type checking):

```bash
uv sync --extra dev
```

## Usage

The framework currently uses a built-in mock server (python-chess). A real MCP
server integration is planned — see `docs/mcp-integration.md`.

```bash
# Play as white against Claude
python -m chess_lmm --color white

# Play as black
python -m chess_lmm --color black

# Random color assignment
python -m chess_lmm --color random

# Custom starting position
python -m chess_lmm --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Choose a specific model (default: claude-sonnet-4-6)
python -m chess_lmm --model claude-opus-4-6

# Custom log directory (default: ./game-logs)
python -m chess_lmm --log-dir ./my-logs

# Verbose logging
python -m chess_lmm -v
```

### In-Game Commands

During your turn, enter a move in SAN (e.g., `e4`, `Nf3`, `O-O`) or LAN
(e.g., `e2e4`, `g1f3`). Other commands:

| Command | Description |
|---------|-------------|
| `/moves` | Show all legal moves |
| `/moves e2` | Show legal moves from a specific square |
| `/board` | Display the current board |
| `/status` | Show game status |
| `/history` | Show move history |
| `/draw offer` | Offer a draw |
| `/draw claim` | Claim a draw (fifty-move / threefold repetition) |
| `/draw accept` | Accept a pending draw offer |
| `/draw decline` | Decline a pending draw offer |
| `/resign` | Resign the game |
| `/help` | Show available commands |

### Game Logs

Each game produces two log files in the log directory:

- `mcp_recording.jsonl` — all MCP tool calls and responses (JSON-lines)
- `llm_interactions.jsonl` — Anthropic API request/response pairs

## Architecture

```
orchestrator.py          CLI entry point, game lifecycle
├── human_player.py      Human player (stdin/stdout commands)
├── llm_agent.py         Claude agentic loop (tool-use cycle)
├── recording.py         JSON-lines logging + board rendering
├── mcp_interface.py     Protocol abstraction (ChessSessionClient)
└── mock_server.py       In-memory chess server (python-chess)
```

The `ChessSessionClient` Protocol is the abstraction boundary. The mock server
satisfies it directly; a real MCP server will be wrapped by an
`McpSessionClient` adapter (see `docs/mcp-integration.md`). Players, recording,
and the orchestrator are server-agnostic.

## Development

```bash
uv run pytest --cov            # Run tests with coverage (147 tests)
uv run mypy src/               # Type check (strict mode)
uv run ruff check src/ tests/  # Lint
uv run ruff format src/ tests/ # Format
```

## Spec

The MCP tool interface follows the chess engine specification in
[code-samples/chess/SPEC.md](https://github.com/nverhaaren/code-samples/blob/master/chess/SPEC.md)
Section 8.

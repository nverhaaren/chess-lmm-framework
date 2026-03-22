# MCP Server Integration Design

How the chess-lmm-framework connects to a real MCP chess server, replacing the
in-process mock.

## Goals

1. Swap `MockChessServer` for a real MCP server with zero changes to the player
   implementations (human_player, llm_agent), recording layer, or game loop.
2. Support both **external** servers (user starts separately) and **managed**
   servers (orchestrator spawns and manages the subprocess).
3. Use the `mcp` Python SDK client — avoid reimplementing the Streamable HTTP
   transport.

## Current Architecture

```
orchestrator.py
  ├── MockChessServer          ← creates shared in-memory game
  │     ├── create_session()   ← returns MockSessionClient (sync)
  │     └── create_session()
  ├── RecordingClient(human_session)
  ├── RecordingClient(llm_session)
  ├── human_turn(human_client, ...)
  └── llm_turn(llm_client, ...)
```

`ChessSessionClient` (Protocol) is the abstraction boundary. Everything above
it is server-agnostic. Everything below it is server-specific.

## Target Architecture

```
orchestrator.py
  ├── --server-url  →  McpServerConnection(url)     [external]
  │       or
  │   --mock        →  MockChessServer()             [mock, for testing]
  │       or
  │   (default)     →  ManagedMcpServer(command)     [managed subprocess]
  │
  ├── server.create_session()   → McpSessionClient   [async]
  ├── server.create_session()   → McpSessionClient
  ├── RecordingClient(human_session)
  ├── RecordingClient(llm_session)
  ├── human_turn(...)           ← unchanged
  └── llm_turn(...)             ← unchanged
```

## Protocol Changes

### ChessServerFactory: async create_session

The current `ChessServerFactory.create_session()` is synchronous. For real MCP,
session creation involves an HTTP roundtrip (the `initialize` handshake that
assigns an `Mcp-Session-Id`). The factory method must become async:

```python
class ChessServerFactory(Protocol):
    async def create_session(self) -> ChessSessionClient: ...
```

This is a **breaking change** to `MockChessServer.create_session()` — it will
need the `async` keyword added. Callers already `await` the result of methods
on the returned client, so the impact is small: just add `await` at the two
`create_session()` call sites in the orchestrator.

### ChessSessionClient: unchanged

The Protocol itself needs no changes. Each method already returns the correct
TypedDict. The `McpSessionClient` implementation translates each method into an
MCP `tools/call` request and deserializes the response.

## McpSessionClient Implementation

### Lifecycle

Each `McpSessionClient` wraps an `mcp.ClientSession` connected via Streamable
HTTP. The SDK requires two nested async context managers:

```python
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

class McpSessionClient:
    def __init__(self, url: str, session_name: str) -> None:
        self._url = url
        self._session_name = session_name
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            streamable_http_client(self._url)
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()  # sends DELETE to end MCP session
```

### Tool Calls

Each `ChessSessionClient` method maps to a single `session.call_tool()`:

```python
async def make_move(self, move: str) -> MakeMoveResult:
    return await self._call_tool("make_move", {"move": move})

async def get_status(self) -> GameStatus:
    return await self._call_tool("get_status", {})
```

The shared `_call_tool` method handles serialization, error translation, and
response parsing:

```python
async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
    assert self._session is not None, "Not connected"
    result = await self._session.call_tool(name, arguments)

    if result.isError:
        # Tool-level errors: content[0].text is a JSON error object
        error_data = json.loads(result.content[0].text)
        raise McpError(
            code=error_data["error"],
            message=error_data["message"],
            detail=error_data.get("detail"),
        )

    # Success: parse structured content or text content as JSON
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)
```

**Design choice — structuredContent vs text:** The MCP spec supports
`outputSchema` on tools, which enables typed `structuredContent` in responses.
If the chess MCP server declares output schemas (recommended), we use
`structuredContent` directly as a dict. Otherwise, fall back to parsing the
text content as JSON. Either way, the returned dict satisfies the TypedDict
protocol.

**Error mapping:** The chess MCP server returns tool errors with
`isError: true` and the error body in `content[0].text` as JSON matching
`{"error": "<code>", "message": "...", "detail": ...}`. The `_call_tool`
method parses this and raises `McpError` so the rest of the framework sees
the same exception type as with the mock.

### Session Identity

The SDK manages `Mcp-Session-Id` headers automatically — the client cannot
read or set them. For logging/correlation purposes, `McpSessionClient` tracks
its own `session_name` (e.g., "human", "llm") which is passed to
`GameRecorder`. This is analogous to the mock's `session_id` field.

## Server Connection Modes

### Mode 1: External Server (`--server-url`)

The server is already running. The orchestrator connects to it:

```
python -m chess_lmm --server-url http://localhost:8000/mcp --color white
```

Implementation:

```python
class McpServerConnection:
    """Factory for sessions to an already-running MCP server."""

    def __init__(self, url: str) -> None:
        self._url = url

    async def create_session(self) -> McpSessionClient:
        client = McpSessionClient(self._url, session_name=f"session-{...}")
        await client.connect()
        return client
```

### Mode 2: Managed Server (default)

The orchestrator starts the server subprocess, waits for readiness, then
connects. On exit (normal or exception), it shuts down the subprocess.

```
python -m chess_lmm --color white
# Implicitly launches the MCP server
```

Implementation sketch:

```python
class ManagedMcpServer:
    """Launches and manages an MCP server subprocess."""

    def __init__(self, command: list[str], port: int = 0) -> None:
        self._command = command
        self._port = port
        self._process: asyncio.subprocess.Process | None = None
        self._url: str | None = None

    async def start(self) -> None:
        """Start the server subprocess and wait for readiness."""
        # If port=0, pick an available port
        if self._port == 0:
            self._port = _find_free_port()

        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            "--port", str(self._port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._url = f"http://localhost:{self._port}/mcp"
        await self._wait_for_ready()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)

    async def create_session(self) -> McpSessionClient:
        assert self._url is not None, "Server not started"
        client = McpSessionClient(self._url, session_name=f"session-{...}")
        await client.connect()
        return client

    async def _wait_for_ready(self, timeout: float = 10.0) -> None:
        """Poll until the server responds to HTTP or a readiness signal."""
        ...
```

**Readiness detection options:**
1. **Stdout sentinel:** Server prints a line like `Listening on port 8000` —
   the manager reads stdout lines until it sees the sentinel.
2. **HTTP poll:** Attempt a GET to the `/mcp` endpoint until it responds.
3. **Both:** Read stdout for port, then HTTP-poll for MCP readiness.

Option 3 is most robust. The managed server should print its port to stdout
(especially if port=0 was used) and then the manager confirms MCP readiness
with a probe.

**Server command configuration:** The default command will be determined by
whatever the MCP server's entry point is. For the C++ engine wrapped in Python:

```
python -m chess_mcp_server --port {port}
```

This can be overridden with `--server-command`:

```
python -m chess_lmm --server-command "python -m chess_mcp_server" --color white
```

### Mode 3: Mock (for testing)

The existing `MockChessServer` remains available for tests and offline use:

```
python -m chess_lmm --mock --color white
```

## Orchestrator Changes

### CLI Arguments

New arguments:

| Argument | Description |
|----------|-------------|
| `--server-url URL` | Connect to an external MCP server |
| `--server-command CMD` | Command to launch a managed server |
| `--mock` | Use the in-process mock server (no MCP) |

`--server-url` and `--mock` are mutually exclusive. If neither is given, the
orchestrator uses managed mode with a default server command.

### Server Selection Logic

```python
async def _create_server(args) -> ChessServerFactory:
    if args.mock:
        return MockChessServer()
    elif args.server_url:
        return McpServerConnection(args.server_url)
    else:
        server = ManagedMcpServer(
            command=shlex.split(args.server_command or DEFAULT_SERVER_CMD),
        )
        await server.start()
        return server
```

### Lifecycle Management

The orchestrator needs to ensure cleanup on all exit paths. Use an
`AsyncExitStack`:

```python
async def run_game(args):
    async with AsyncExitStack() as stack:
        server = await _create_server(args)
        if isinstance(server, ManagedMcpServer):
            stack.push_async_callback(server.stop)

        human_session = await server.create_session()
        llm_session = await server.create_session()
        if isinstance(human_session, McpSessionClient):
            stack.push_async_callback(human_session.close)
            stack.push_async_callback(llm_session.close)

        # ... rest of game loop unchanged ...
```

## Error Handling

### Connection Failures

If the MCP server is unreachable or the initialize handshake fails, the SDK
raises `McpError` or `httpx` connection errors. The orchestrator should catch
these at startup and report a clear message:

```
Error: Could not connect to MCP server at http://localhost:8000/mcp
```

### Mid-Game Disconnection

If the server crashes or the connection drops during a game, `call_tool` will
raise. The game loop already catches `McpError` — the orchestrator should
detect transport-level errors and surface them distinctly from game-logic
errors:

```
Error: Lost connection to MCP server. Game state may be lost.
```

No automatic reconnection — chess game state is server-side and
non-recoverable if the server process dies.

### Session Expiry

The MCP SDK has a known issue (#1676) where it doesn't auto-reinitialize on
HTTP 404 (expired session). If this affects us, the `McpSessionClient` can
catch it and attempt one reconnect. For chess games (single session per game,
short-lived), this is unlikely to be a practical issue.

## Testing Strategy

### Unit Tests

- `McpSessionClient._call_tool` with mocked `ClientSession`: verify tool name
  and arguments are passed correctly, verify error parsing, verify response
  deserialization.
- `McpServerConnection.create_session`: verify it creates and connects a
  client.
- `ManagedMcpServer`: verify subprocess launch, readiness polling, and
  shutdown.

### Integration Tests

- Start a real MCP server (the chess MCP server) in a subprocess, connect with
  `McpServerConnection`, play a short game, verify the full flow end-to-end.
- These tests require the MCP server to be built first, so they'll be gated
  behind a marker (`@pytest.mark.integration`).

### Existing Tests

All existing tests continue to use `MockChessServer` directly. No changes
needed.

## Implementation Order

1. **Make `create_session` async** — small breaking change to Protocol,
   MockChessServer, and orchestrator. Do this first as a standalone PR so the
   interface is ready.
2. **Implement `McpSessionClient`** — the client wrapper with `_call_tool`,
   `connect`, `close`. Unit tests with mocked `ClientSession`.
3. **Implement `McpServerConnection`** (external mode) — factory that creates
   `McpSessionClient` instances. Add `--server-url` to orchestrator.
4. **Implement `ManagedMcpServer`** (managed mode) — subprocess lifecycle.
   Add `--server-command` and `--mock` flags.
5. **Integration tests** — once the real MCP server exists.

Steps 2-3 can be a single PR. Step 4 is a separate PR since it involves
subprocess management which is more complex to test.

## Dependencies

- `mcp` package (already in project dependencies)
- `httpx` (transitive dependency of `mcp`)
- No new direct dependencies needed

## Open Questions

1. **Error response format:** The design assumes tool errors are returned as
   JSON in `content[0].text` with `{"error": "...", "message": "..."}`. This
   must match whatever the chess MCP server actually produces. Coordinate with
   the server implementation.
2. **structuredContent support:** If the chess MCP server declares
   `outputSchema` on its tools, `structuredContent` gives us typed dicts
   directly and we skip JSON parsing. Worth doing but not required for v1.
3. **Default server command:** What's the entry point for the chess MCP
   server? Needs to be determined when the server is built. Likely
   `python -m chess_mcp_server` or similar.
4. **Port assignment:** Should the managed server always use a random port
   (safest), or allow the user to specify one?
5. **Health check endpoint:** Does the chess MCP server expose a readiness
   endpoint, or do we rely on the MCP initialize handshake as the readiness
   signal?

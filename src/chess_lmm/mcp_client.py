"""MCP client adapter: connects to a real MCP chess server.

Wraps the ``mcp`` SDK's ClientSession to satisfy the ChessSessionClient
protocol. Each McpSessionClient holds one SSE connection; McpServerConnection
is the factory that creates them.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast

from chess_lmm.types import (
    BoardResult,
    CreateGameResult,
    DeclineDrawResult,
    DoneResult,
    ExportResult,
    GameStatus,
    HistoryResult,
    JoinGameResult,
    LegalMovesResult,
    MakeMoveResult,
    McpError,
    MessagesResult,
    OfferDrawResult,
    SendMessageResult,
)

if TYPE_CHECKING:
    from mcp import ClientSession  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def _import_mcp() -> tuple[Any, Any]:
    """Import MCP SDK components, raising McpError if not installed.

    Returns (ClientSession class, sse_client function).
    """
    try:
        import mcp
        import mcp.client.sse  # type: ignore[import-not-found]
    except ImportError:
        raise McpError(
            "missing_dependency",
            "The 'mcp' package is required for MCP server connections. "
            "Install with: uv sync --extra mcp",
        ) from None
    return mcp.ClientSession, mcp.client.sse.sse_client


class McpSessionClient:
    """Per-session MCP client wrapping a ClientSession.

    Each instance holds one SSE connection to the chess MCP server.
    Call connect() before use and close() when done.
    """

    def __init__(self, url: str, session_id: str) -> None:
        self._url = url
        self._session_id = session_id
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    async def connect(self) -> None:
        """Establish SSE connection and initialize the MCP session."""
        client_session_cls, sse_client_fn = _import_mcp()
        self._exit_stack = AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client_fn(self._url)
        )
        self._session = await self._exit_stack.enter_async_context(
            client_session_cls(read_stream, write_stream)
        )
        await self._session.initialize()  # type: ignore[union-attr]
        logger.info("MCP session %s connected to %s", self._session_id, self._url)

    async def close(self) -> None:
        """Tear down the SSE connection."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            logger.info("MCP session %s closed", self._session_id)

    # --- Core tool call ---

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call an MCP tool and return the parsed JSON result.

        Raises McpError if the server returns isError=True or on protocol errors.
        """
        if self._session is None:
            raise McpError(
                "not_connected",
                "MCP session not connected. Call connect() first.",
            )

        result = await self._session.call_tool(tool_name, arguments or {})

        if not result.content:
            raise McpError("internal_error", f"Empty response from tool '{tool_name}'.")

        first = result.content[0]
        if not hasattr(first, "text"):
            raise McpError(
                "internal_error",
                f"Non-text content from tool '{tool_name}'.",
            )

        try:
            data: dict[str, Any] = json.loads(first.text)
        except (json.JSONDecodeError, TypeError) as e:
            raise McpError(
                "internal_error",
                f"Invalid JSON from tool '{tool_name}': {e}",
            ) from e

        if result.isError:
            error_code = data.get("error", "server_error")
            error_message = data.get("message", str(data))
            error_detail = data.get("detail")
            raise McpError(error_code, error_message, detail=error_detail)

        return data

    # --- Game management (Section 8.5.1) ---

    async def create_game(
        self,
        *,
        fen: str | None = None,
        history: list[str] | None = None,
    ) -> CreateGameResult:
        args: dict[str, Any] = {}
        if fen is not None:
            args["fen"] = fen
        if history is not None:
            args["history"] = history
        return cast(CreateGameResult, await self._call_tool("create_game", args))

    async def join_game(
        self,
        color: str,
        *,
        name: str | None = None,
    ) -> JoinGameResult:
        args: dict[str, Any] = {"color": color}
        if name is not None:
            args["name"] = name
        return cast(JoinGameResult, await self._call_tool("join_game", args))

    async def export_game(self, *, format: str = "pgn") -> ExportResult:
        return cast(
            ExportResult, await self._call_tool("export_game", {"format": format})
        )

    async def done(self) -> DoneResult:
        return cast(DoneResult, await self._call_tool("done"))

    # --- Query tools (Section 8.5.2) ---

    async def get_board(self) -> BoardResult:
        return cast(BoardResult, await self._call_tool("get_board"))

    async def get_status(self) -> GameStatus:
        return cast(GameStatus, await self._call_tool("get_status"))

    async def get_legal_moves(
        self,
        *,
        square: str | None = None,
        format: str = "both",
    ) -> LegalMovesResult:
        args: dict[str, Any] = {"format": format}
        if square is not None:
            args["square"] = square
        return cast(LegalMovesResult, await self._call_tool("get_legal_moves", args))

    async def get_history(self, *, format: str = "san") -> HistoryResult:
        return cast(
            HistoryResult, await self._call_tool("get_history", {"format": format})
        )

    async def get_messages(self, *, clear: bool = True) -> MessagesResult:
        return cast(
            MessagesResult,
            await self._call_tool("get_messages", {"clear": clear}),
        )

    # --- Action tools (Section 8.5.3) ---

    async def make_move(self, move: str) -> MakeMoveResult:
        return cast(MakeMoveResult, await self._call_tool("make_move", {"move": move}))

    async def claim_draw(self) -> GameStatus:
        return cast(GameStatus, await self._call_tool("claim_draw"))

    async def offer_draw(self) -> OfferDrawResult:
        return cast(OfferDrawResult, await self._call_tool("offer_draw"))

    async def accept_draw(self) -> GameStatus:
        return cast(GameStatus, await self._call_tool("accept_draw"))

    async def decline_draw(self) -> DeclineDrawResult:
        return cast(DeclineDrawResult, await self._call_tool("decline_draw"))

    async def resign(self) -> GameStatus:
        return cast(GameStatus, await self._call_tool("resign"))

    async def send_message(self, text: str) -> SendMessageResult:
        return cast(
            SendMessageResult,
            await self._call_tool("send_message", {"text": text}),
        )


class McpServerConnection:
    """Factory that creates per-session MCP clients connected to a chess server.

    Satisfies the ChessServerFactory protocol.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._session_counter = 0
        self._clients: list[McpSessionClient] = []

    async def create_session(self) -> McpSessionClient:
        """Create a new connected MCP session client."""
        self._session_counter += 1
        session_id = f"mcp-session-{self._session_counter}"
        client = McpSessionClient(self._url, session_id)
        await client.connect()
        self._clients.append(client)
        return client

    async def close_all(self) -> None:
        """Close all sessions created by this factory."""
        for client in self._clients:
            try:
                await client.close()
            except Exception:
                logger.warning(
                    "Error closing MCP session %s",
                    client.session_id,
                    exc_info=True,
                )
        self._clients.clear()

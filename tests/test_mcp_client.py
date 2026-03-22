"""Tests for MCP client adapter.

All MCP SDK types are mocked — these tests do not require the ``mcp`` package.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chess_lmm.mcp_client import McpServerConnection, McpSessionClient
from chess_lmm.types import McpError

# --- Mock MCP SDK types ---


class MockTextContent:
    """Mimics mcp TextContent."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockCallToolResult:
    """Mimics mcp CallToolResult."""

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [MockTextContent(text)]
        self.isError = is_error


class MockEmptyResult:
    """CallToolResult with no content."""

    def __init__(self) -> None:
        self.content: list[Any] = []
        self.isError = False


class MockNonTextResult:
    """CallToolResult with non-text content."""

    def __init__(self) -> None:
        self.content = [MagicMock(spec=[])]  # No .text attribute
        self.isError = False


# --- Fixtures ---


@pytest.fixture
def mcp_client() -> McpSessionClient:
    """McpSessionClient with a mocked session (bypasses connect())."""
    client = McpSessionClient("http://localhost:8000/sse", "test-session-1")
    client._session = MagicMock()  # type: ignore[assignment]
    return client


# --- TestCallTool ---


class TestCallTool:
    """Tests for the core _call_tool method."""

    async def test_successful_call(self, mcp_client: McpSessionClient) -> None:
        payload = {
            "game_id": "g1",
            "fen": "startpos",
            "game_status": "awaiting_players",
        }
        mock_result = MockCallToolResult(json.dumps(payload))
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        result = await mcp_client._call_tool("create_game", {"fen": None})
        assert result == payload

    async def test_error_response_raises_mcp_error(
        self, mcp_client: McpSessionClient
    ) -> None:
        error_body = {"error": "illegal_move", "message": "Pawn cannot reach e5"}
        mock_result = MockCallToolResult(json.dumps(error_body), is_error=True)
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        with pytest.raises(McpError) as exc_info:
            await mcp_client._call_tool("make_move", {"move": "e5"})
        assert exc_info.value.code == "illegal_move"
        assert "Pawn cannot reach e5" in exc_info.value.message

    async def test_error_with_detail(self, mcp_client: McpSessionClient) -> None:
        error_body = {
            "error": "internal_error",
            "message": "Unexpected",
            "detail": "stack trace here",
        }
        mock_result = MockCallToolResult(json.dumps(error_body), is_error=True)
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        with pytest.raises(McpError) as exc_info:
            await mcp_client._call_tool("get_status")
        assert exc_info.value.code == "internal_error"
        assert exc_info.value.detail == "stack trace here"

    async def test_empty_content_raises(self, mcp_client: McpSessionClient) -> None:
        mock_result = MockEmptyResult()
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        with pytest.raises(McpError, match="Empty response"):
            await mcp_client._call_tool("get_board")

    async def test_non_text_content_raises(self, mcp_client: McpSessionClient) -> None:
        mock_result = MockNonTextResult()
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        with pytest.raises(McpError, match="Non-text content"):
            await mcp_client._call_tool("get_board")

    async def test_invalid_json_raises(self, mcp_client: McpSessionClient) -> None:
        mock_result = MockCallToolResult("not valid json")
        mcp_client._session.call_tool = AsyncMock(return_value=mock_result)  # type: ignore[union-attr]

        with pytest.raises(McpError, match="Invalid JSON"):
            await mcp_client._call_tool("get_status")

    async def test_not_connected_raises(self) -> None:
        client = McpSessionClient("http://localhost:8000/sse", "test")
        # Don't connect — _session is None
        with pytest.raises(McpError, match="not connected"):
            await client._call_tool("get_status")


# --- TestProtocolMethods ---


class TestProtocolMethods:
    """Verify each protocol method calls _call_tool with correct args."""

    async def test_create_game_no_args(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"game_id": "g1"}))
        )
        await mcp_client.create_game()
        mcp_client._session.call_tool.assert_called_once_with("create_game", {})  # type: ignore[union-attr]

    async def test_create_game_with_fen(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"game_id": "g1"}))
        )
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        await mcp_client.create_game(fen=fen)
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args[0] == "create_game"
        assert "fen" in args[1]

    async def test_join_game(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"assigned_color": "white"}))
        )
        await mcp_client.join_game("white")
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args == ("join_game", {"color": "white"})

    async def test_join_game_with_name(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"assigned_color": "white"}))
        )
        await mcp_client.join_game("white", name="Alice")
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args == ("join_game", {"color": "white", "name": "Alice"})

    async def test_make_move(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"server_state": "ongoing"}))
        )
        await mcp_client.make_move("e4")
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args == ("make_move", {"move": "e4"})

    async def test_get_status(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"server_state": "ongoing"}))
        )
        await mcp_client.get_status()
        mcp_client._session.call_tool.assert_called_once_with("get_status", {})  # type: ignore[union-attr]

    async def test_get_legal_moves_with_square(
        self, mcp_client: McpSessionClient
    ) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"moves": [], "count": 0}))
        )
        await mcp_client.get_legal_moves(square="e2")
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args == ("get_legal_moves", {"format": "both", "square": "e2"})

    async def test_resign(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"server_state": "game_over"}))
        )
        await mcp_client.resign()
        mcp_client._session.call_tool.assert_called_once_with("resign", {})  # type: ignore[union-attr]

    async def test_offer_draw(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"offered": True}))
        )
        await mcp_client.offer_draw()
        mcp_client._session.call_tool.assert_called_once_with("offer_draw", {})  # type: ignore[union-attr]

    async def test_send_message(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(json.dumps({"sent": True}))
        )
        await mcp_client.send_message("hello")
        args = mcp_client._session.call_tool.call_args[0]  # type: ignore[union-attr]
        assert args == ("send_message", {"text": "hello"})

    async def test_done(self, mcp_client: McpSessionClient) -> None:
        mcp_client._session.call_tool = AsyncMock(  # type: ignore[union-attr]
            return_value=MockCallToolResult(
                json.dumps({"acknowledged": True, "clients_remaining": 1})
            )
        )
        result = await mcp_client.done()
        assert result["acknowledged"] is True
        mcp_client._session.call_tool.assert_called_once_with("done", {})  # type: ignore[union-attr]


# --- TestMcpServerConnection ---


class TestMcpServerConnection:
    """Tests for the factory class."""

    async def test_create_session_increments_id(self) -> None:
        conn = McpServerConnection("http://localhost:8000/sse")
        with patch.object(McpSessionClient, "connect", new_callable=AsyncMock):
            s1 = await conn.create_session()
            s2 = await conn.create_session()
        assert s1.session_id != s2.session_id
        assert s1.session_id == "mcp-session-1"
        assert s2.session_id == "mcp-session-2"

    async def test_create_session_calls_connect(self) -> None:
        conn = McpServerConnection("http://localhost:8000/sse")
        with patch.object(
            McpSessionClient, "connect", new_callable=AsyncMock
        ) as mock_connect:
            await conn.create_session()
        mock_connect.assert_called_once()

    async def test_close_all(self) -> None:
        conn = McpServerConnection("http://localhost:8000/sse")
        with patch.object(McpSessionClient, "connect", new_callable=AsyncMock):
            await conn.create_session()
            await conn.create_session()

        with patch.object(
            McpSessionClient, "close", new_callable=AsyncMock
        ) as mock_close:
            await conn.close_all()
        assert mock_close.call_count == 2

    async def test_close_all_tolerates_errors(self) -> None:
        conn = McpServerConnection("http://localhost:8000/sse")
        with patch.object(McpSessionClient, "connect", new_callable=AsyncMock):
            await conn.create_session()
            await conn.create_session()

        with patch.object(
            McpSessionClient,
            "close",
            new_callable=AsyncMock,
            side_effect=OSError("boom"),
        ):
            # Should not raise
            await conn.close_all()


# --- TestOrchestrator CLI flags ---


class TestServerFlags:
    """Tests for --server-url and --mock CLI flags."""

    def test_default_is_no_server_url(self) -> None:
        from chess_lmm.orchestrator import parse_args

        args = parse_args([])
        assert args.server_url is None
        assert args.mock is False

    def test_server_url(self) -> None:
        from chess_lmm.orchestrator import parse_args

        args = parse_args(["--server-url", "http://localhost:8000/sse"])
        assert args.server_url == "http://localhost:8000/sse"

    def test_mock_flag(self) -> None:
        from chess_lmm.orchestrator import parse_args

        args = parse_args(["--mock"])
        assert args.mock is True

    def test_server_url_and_mock_mutually_exclusive(self) -> None:
        from chess_lmm.orchestrator import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--server-url", "http://localhost:8000/sse", "--mock"])

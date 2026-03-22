"""Comprehensive tests for the mock chess server.

Covers: state machine transitions, moves, draw flow, session isolation,
error codes, checkmate, stalemate, and spec edge cases.
"""

from __future__ import annotations

import pytest

from chess_lmm.mock_server import MockChessServer
from chess_lmm.types import McpError

from .conftest import INITIAL_FEN, STALEMATE_FEN

# --- Fixtures ---


@pytest.fixture
def server() -> MockChessServer:
    """Fresh mock server with no game."""
    return MockChessServer()


async def _setup_game(
    server: MockChessServer,
    *,
    fen: str | None = None,
    history: list[str] | None = None,
) -> tuple:
    """Create game and join two players. Returns (white, black)."""
    white = await server.create_session()
    black = await server.create_session()
    await white.create_game(fen=fen, history=history)
    await white.join_game("white")
    await black.join_game("black")
    return white, black


# --- State machine tests ---


class TestStateMachine:
    """Test server state transitions."""

    async def test_initial_state_is_no_game(self, server: MockChessServer) -> None:
        assert server.game.server_state == "no_game"

    async def test_create_game_transitions_to_awaiting(
        self, server: MockChessServer
    ) -> None:
        client = await server.create_session()
        result = await client.create_game()
        assert result["game_status"] == "awaiting_players"
        assert server.game.server_state == "awaiting_players"

    async def test_first_join_stays_awaiting(self, server: MockChessServer) -> None:
        client = await server.create_session()
        await client.create_game()
        result = await client.join_game("white")
        assert result["game_status"] == "awaiting_players"
        assert result["assigned_color"] == "white"

    async def test_second_join_transitions_to_ongoing(
        self, server: MockChessServer
    ) -> None:
        white = await server.create_session()
        black = await server.create_session()
        await white.create_game()
        await white.join_game("white")
        result = await black.join_game("black")
        assert result["game_status"] == "ongoing"
        assert server.game.server_state == "ongoing"

    async def test_cannot_create_game_twice(self, server: MockChessServer) -> None:
        client = await server.create_session()
        await client.create_game()
        with pytest.raises(McpError, match="wrong_state"):
            await client.create_game()

    async def test_cannot_join_without_game(self, server: MockChessServer) -> None:
        client = await server.create_session()
        with pytest.raises(McpError, match="no_active_game"):
            await client.join_game("white")


class TestCreateGame:
    """Test game creation with various parameters."""

    async def test_default_fen(self, server: MockChessServer) -> None:
        client = await server.create_session()
        result = await client.create_game()
        assert result["fen"] == INITIAL_FEN
        assert result["time_control"] is None
        assert result["game_id"]  # non-empty

    async def test_custom_fen(self, server: MockChessServer) -> None:
        client = await server.create_session()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = await client.create_game(fen=fen)
        assert result["fen"] == fen

    async def test_invalid_fen(self, server: MockChessServer) -> None:
        client = await server.create_session()
        with pytest.raises(McpError) as exc_info:
            await client.create_game(fen="not-a-fen")
        assert exc_info.value.code == "invalid_fen"

    async def test_fen_with_invalid_position(self, server: MockChessServer) -> None:
        # Two white kings
        client = await server.create_session()
        with pytest.raises(McpError) as exc_info:
            await client.create_game(fen="KK6/8/8/8/8/8/8/8 w - - 0 1")
        assert exc_info.value.code == "invalid_fen"

    async def test_history_replay(self, server: MockChessServer) -> None:
        client = await server.create_session()
        result = await client.create_game(history=["e4", "e5", "Nf3"])
        # After 1. e4 e5 2. Nf3, it's Black's move
        assert "b KQkq" in result["fen"]

    async def test_invalid_history(self, server: MockChessServer) -> None:
        client = await server.create_session()
        with pytest.raises(McpError) as exc_info:
            await client.create_game(history=["e4", "e4"])  # e4 twice is illegal
        assert exc_info.value.code == "invalid_history"

    async def test_fen_plus_history(self, server: MockChessServer) -> None:
        """Create from custom FEN and replay history on top."""
        client = await server.create_session()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = await client.create_game(fen=fen, history=["e5"])
        assert "w KQkq" in result["fen"]  # White's move after e5


class TestJoinGame:
    """Test join_game behavior."""

    async def test_join_white(self, server: MockChessServer) -> None:
        client = await server.create_session()
        await client.create_game()
        result = await client.join_game("white")
        assert result["assigned_color"] == "white"

    async def test_join_black(self, server: MockChessServer) -> None:
        white = await server.create_session()
        black = await server.create_session()
        await white.create_game()
        await white.join_game("white")
        result = await black.join_game("black")
        assert result["assigned_color"] == "black"

    async def test_join_random(self, server: MockChessServer) -> None:
        client = await server.create_session()
        await client.create_game()
        result = await client.join_game("random")
        assert result["assigned_color"] in ("white", "black")

    async def test_color_taken(self, server: MockChessServer) -> None:
        c1 = await server.create_session()
        c2 = await server.create_session()
        await c1.create_game()
        await c1.join_game("white")
        with pytest.raises(McpError, match="color_taken"):
            await c2.join_game("white")

    async def test_already_joined(self, server: MockChessServer) -> None:
        client = await server.create_session()
        await client.create_game()
        await client.join_game("white")
        with pytest.raises(McpError, match="already_joined"):
            await client.join_game("black")


class TestMakeMove:
    """Test move making."""

    async def test_simple_move_san(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.make_move("e4")
        assert result["move_played"]["san"] == "e4"
        assert result["move_played"]["lan"] == "e2e4"
        assert result["turn"] == "black"

    async def test_simple_move_lan(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.make_move("e2e4")
        assert result["move_played"]["san"] == "e4"

    async def test_not_your_turn(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError, match="not_your_turn"):
            await black.make_move("e5")

    async def test_alternating_turns(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        result = await black.make_move("e5")
        assert result["turn"] == "white"

    async def test_invalid_move_format(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError) as exc_info:
            await white.make_move("xyz")
        assert exc_info.value.code == "invalid_format"

    async def test_illegal_move(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError) as exc_info:
            await white.make_move("e2e5")  # Pawn can't jump 3 squares
        assert exc_info.value.code == "illegal_move"

    async def test_capture(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await black.make_move("d5")
        result = await white.make_move("exd5")
        assert result["move_played"]["san"] == "exd5"

    async def test_castling_san(self, server: MockChessServer) -> None:
        """Test kingside castling via SAN."""
        white, black = await _setup_game(
            server, history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]
        )
        result = await white.make_move("O-O")
        assert result["move_played"]["san"] == "O-O"
        assert result["move_played"]["lan"] == "e1g1"

    async def test_castling_lan(self, server: MockChessServer) -> None:
        """Test kingside castling via LAN."""
        white, black = await _setup_game(
            server, history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]
        )
        result = await white.make_move("e1g1")
        assert result["move_played"]["san"] == "O-O"

    async def test_promotion(self, server: MockChessServer) -> None:
        """Test pawn promotion."""
        # Set up a position with a pawn about to promote
        fen = "8/P7/8/8/8/8/8/k1K5 w - - 0 1"
        white, black = await _setup_game(server, fen=fen)
        result = await white.make_move("a8=Q")
        assert "Q" in result["move_played"]["san"]

    async def test_san_normalization_annotation_stripped(
        self, server: MockChessServer
    ) -> None:
        """Move with assessment glyph should be accepted after stripping."""
        white, black = await _setup_game(server)
        result = await white.make_move("e4!")
        assert result["move_played"]["san"] == "e4"

    async def test_san_normalization_double_annotation(
        self, server: MockChessServer
    ) -> None:
        """Double assessment glyph (!!, ??) should be stripped."""
        white, black = await _setup_game(server)
        result = await white.make_move("e4!!")
        assert result["move_played"]["san"] == "e4"

    async def test_san_normalization_digit_zero_castling(
        self, server: MockChessServer
    ) -> None:
        """Digit-zero castling (0-0) should be converted to O-O."""
        white, black = await _setup_game(
            server, history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]
        )
        result = await white.make_move("0-0")
        assert result["move_played"]["san"] == "O-O"

    async def test_san_normalization_nag_stripped(
        self, server: MockChessServer
    ) -> None:
        """NAG ($N) should be stripped."""
        white, black = await _setup_game(server)
        result = await white.make_move("e4 $1")
        assert result["move_played"]["san"] == "e4"

    async def test_ambiguous_move_error(self, server: MockChessServer) -> None:
        """Ambiguous SAN should produce ambiguous_move error code."""
        # Two white knights on b1 and f1 can both go to d2 — "Nd2" is ambiguous
        fen = "7k/8/8/8/8/8/8/1N1K1N2 w - - 0 1"
        white, black = await _setup_game(server, fen=fen)
        with pytest.raises(McpError) as exc_info:
            await white.make_move("Nd2")
        assert exc_info.value.code == "ambiguous_move"


class TestCheckmate:
    """Test checkmate detection."""

    async def test_scholars_mate(self, server: MockChessServer) -> None:
        """1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"""
        white, black = await _setup_game(server)
        moves = [
            ("e4", "e5"),
            ("Qh5", "Nc6"),
            ("Bc4", "Nf6"),
        ]
        for w_move, b_move in moves:
            await white.make_move(w_move)
            await black.make_move(b_move)

        result = await white.make_move("Qxf7")
        assert result["game_status"] == "checkmate"
        # White delivered checkmate, White wins
        assert result["result"] == "1-0"
        assert "Checkmate" in (result.get("termination_reason") or "")

    async def test_checkmate_from_fen(self, server: MockChessServer) -> None:
        """Fool's mate position — checkmate already delivered."""
        # After 1. f3 e5 2. g4 Qh4#
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        white, black = await _setup_game(server, fen=fen)
        # White is in checkmate — game should reflect this
        status = await white.get_status()
        assert status["game_status"] == "checkmate"

    async def test_no_moves_after_checkmate(self, server: MockChessServer) -> None:
        """Cannot make moves after checkmate."""
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        white, black = await _setup_game(server, fen=fen)
        with pytest.raises(McpError, match="game_over"):
            await white.make_move("e4")


class TestStalemate:
    """Test stalemate detection."""

    async def test_stalemate_lone_king(self, server: MockChessServer) -> None:
        """Section 9.2.1: k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"""
        white, black = await _setup_game(server, fen=STALEMATE_FEN)
        status = await black.get_status()
        assert status["game_status"] == "stalemate"
        assert status["result"] == "1/2-1/2"


class TestCheck:
    """Test check detection."""

    async def test_check_status(self, server: MockChessServer) -> None:
        """After a move that delivers check, status should say 'check'."""
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await black.make_move("f5")
        result = await white.make_move("Qh5")  # Check!
        assert result["is_check"] is True
        assert result["game_status"] == "check"


class TestDrawFlow:
    """Test draw offer/accept/decline/claim flow."""

    async def test_offer_and_accept_draw(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")

        # White offers draw
        offer_result = await white.offer_draw()
        assert offer_result["offered"] is True

        # Black sees the offer
        status = await black.get_status()
        assert status["draw_offered"] is True

        # White does NOT see a draw offered to them
        w_status = await white.get_status()
        assert w_status["draw_offered"] is False

        # Black accepts
        result = await black.accept_draw()
        assert result["game_status"] == "draw_agreement"
        assert result["result"] == "1/2-1/2"

    async def test_offer_and_decline_draw(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")

        await white.offer_draw()
        result = await black.decline_draw()
        assert result["declined"] is True

        # No longer offered
        status = await black.get_status()
        assert status["draw_offered"] is False

    async def test_double_offer(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await white.offer_draw()
        with pytest.raises(McpError, match="already_offered"):
            await white.offer_draw()

    async def test_accept_without_offer(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        with pytest.raises(McpError, match="no_pending_offer"):
            await black.accept_draw()

    async def test_decline_without_offer(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        with pytest.raises(McpError, match="no_pending_offer"):
            await black.decline_draw()

    async def test_move_blocked_by_pending_draw_offer(
        self, server: MockChessServer
    ) -> None:
        """Must accept or decline draw before moving."""
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await white.offer_draw()  # White offers
        # Black tries to move without responding to draw offer
        with pytest.raises(McpError, match="pending_draw_offer"):
            await black.make_move("e5")

    async def test_draw_offer_withdrawn_on_move(self, server: MockChessServer) -> None:
        """Moving player's draw offer is implicitly withdrawn when they move.

        Per spec: 'Any pending draw offer from the moving player is
        implicitly withdrawn' (Section 8.5.3, make_move postconditions).
        """
        white, black = await _setup_game(server)

        # White offers a draw then moves — offer should be withdrawn
        await white.offer_draw()

        # Black sees the offer
        b_status = await black.get_status()
        assert b_status["draw_offered"] is True

        # White moves (white can still move — the pending_draw_offer check
        # only blocks the player who *received* the offer, not the offerer)
        await white.make_move("e4")

        # White's draw offer should be withdrawn after white moved
        b_status2 = await black.get_status()
        assert b_status2["draw_offered"] is False

    async def test_claim_draw_not_available(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError, match="draw_not_available"):
            await white.claim_draw()


class TestResign:
    """Test resignation."""

    async def test_white_resigns(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.resign()
        assert result["game_status"] == "resigned"
        assert result["result"] == "0-1"
        assert "White resigns" in (result.get("termination_reason") or "")

    async def test_black_resigns(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await black.resign()
        assert result["game_status"] == "resigned"
        assert result["result"] == "1-0"
        assert "Black resigns" in (result.get("termination_reason") or "")


class TestDone:
    """Test done signaling."""

    async def test_done_after_game_over(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.resign()
        result = await white.done()
        assert result["acknowledged"] is True
        assert result["clients_remaining"] == 1
        result2 = await black.done()
        assert result2["clients_remaining"] == 0

    async def test_done_before_game_over(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError, match=r"game_not_over"):
            await white.done()

    async def test_done_twice(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.resign()
        await white.done()
        with pytest.raises(McpError, match="already_done"):
            await white.done()


class TestQueryTools:
    """Test get_board, get_status, get_legal_moves, get_history."""

    async def test_get_board(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_board()
        assert result["fen"] == INITIAL_FEN

    async def test_get_board_not_joined(self, server: MockChessServer) -> None:
        c1 = await server.create_session()
        c2 = await server.create_session()
        await c1.create_game()
        await c1.join_game("white")
        # c2 hasn't joined
        with pytest.raises(McpError, match="not_joined"):
            await c2.get_board()

    async def test_get_status_ongoing(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        status = await white.get_status()
        assert status["server_state"] == "ongoing"
        assert status["turn"] == "white"
        assert status["is_check"] is False
        assert status["result"] is None

    async def test_get_legal_moves_initial(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_legal_moves()
        assert result["count"] == 20  # 16 pawn + 4 knight moves
        # Should be sorted by LAN
        lans = [m["lan"] for m in result["moves"]]
        assert lans == sorted(lans)

    async def test_get_legal_moves_by_square(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_legal_moves(square="e2")
        assert result["count"] == 2  # e3, e4
        lans = [m["lan"] for m in result["moves"]]
        assert "e2e3" in lans
        assert "e2e4" in lans

    async def test_get_legal_moves_invalid_square(
        self, server: MockChessServer
    ) -> None:
        white, black = await _setup_game(server)
        with pytest.raises(McpError, match="invalid_square"):
            await white.get_legal_moves(square="z9")

    async def test_get_legal_moves_san_format(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_legal_moves(format="san")
        for move in result["moves"]:
            assert "san" in move
            assert "lan" not in move

    async def test_get_legal_moves_lan_format(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_legal_moves(format="lan")
        for move in result["moves"]:
            assert "lan" in move
            assert "san" not in move

    async def test_get_history_empty(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.get_history()
        assert result["moves"] == []
        assert result["total_half_moves"] == 0

    async def test_get_history_after_moves(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await black.make_move("e5")

        result = await white.get_history()
        assert result["total_half_moves"] == 2
        assert len(result["moves"]) == 1
        entry = result["moves"][0]
        assert entry["move_number"] == 1
        assert entry["white"] is not None
        assert entry["white"]["san"] == "e4"
        assert entry["black"] is not None
        assert entry["black"]["san"] == "e5"

    async def test_get_history_incomplete_move(self, server: MockChessServer) -> None:
        """After White moves, Black hasn't moved yet."""
        white, black = await _setup_game(server)
        await white.make_move("e4")

        result = await white.get_history()
        assert result["total_half_moves"] == 1
        entry = result["moves"][0]
        assert entry["white"] is not None
        assert entry["black"] is None

    async def test_get_history_both_format(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        result = await white.get_history(format="both")
        assert result["moves"][0]["white"] is not None
        half = result["moves"][0]["white"]
        assert "san" in half
        assert "lan" in half

    async def test_get_history_from_custom_fen(self, server: MockChessServer) -> None:
        """History should start from the FEN's fullmove number."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 5"
        white, black = await _setup_game(server, fen=fen)
        await black.make_move("e5")

        result = await white.get_history()
        assert result["moves"][0]["move_number"] == 5

    async def test_export_fen(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        result = await white.export_game(format="fen")
        assert result["content"] == INITIAL_FEN


class TestSessionIsolation:
    """Test that sessions are properly isolated."""

    async def test_different_session_ids(self, server: MockChessServer) -> None:
        c1 = await server.create_session()
        c2 = await server.create_session()
        assert c1.session_id != c2.session_id

    async def test_draw_offered_only_to_opponent(self, server: MockChessServer) -> None:
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await white.offer_draw()

        # Black sees the offer
        b_status = await black.get_status()
        assert b_status["draw_offered"] is True

        # White does not see it as offered to them
        w_status = await white.get_status()
        assert w_status["draw_offered"] is False


class TestEdgeCases:
    """Spec edge cases and boundary conditions."""

    async def test_en_passant(self, server: MockChessServer) -> None:
        """Test en passant capture."""
        white, black = await _setup_game(server)
        await white.make_move("e4")
        await black.make_move("d5")
        await white.make_move("e5")
        await black.make_move("f5")  # Double push next to white pawn
        result = await white.make_move("exf6")  # En passant
        assert result["move_played"]["san"] == "exf6"

    async def test_start_from_check_position(self, server: MockChessServer) -> None:
        """Game started from a FEN where the side to move is in check."""
        # Position: Black king in check from white queen
        fen = "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1"
        white, black = await _setup_game(server, fen=fen)
        status = await black.get_status()
        assert status["game_status"] == "check"
        assert status["is_check"] is True

    async def test_insufficient_material(self, server: MockChessServer) -> None:
        """K vs K is insufficient material."""
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        white, black = await _setup_game(server, fen=fen)
        status = await white.get_status()
        assert status["insufficient_material"] is True

    async def test_resign_not_turn_gated(self, server: MockChessServer) -> None:
        """Either player can resign regardless of whose turn it is."""
        white, black = await _setup_game(server)
        # It's white's turn, but black can resign
        result = await black.resign()
        assert result["game_status"] == "resigned"

    async def test_offer_draw_not_turn_gated(self, server: MockChessServer) -> None:
        """Either player can offer a draw regardless of turn."""
        white, black = await _setup_game(server)
        # It's white's turn, but black can offer draw
        result = await black.offer_draw()
        assert result["offered"] is True

    async def test_history_replay_with_fen(self, server: MockChessServer) -> None:
        """Create game with FEN and history, verify the resulting position."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        white, black = await _setup_game(server, fen=fen, history=["e5", "Nf3"])
        status = await white.get_status()
        assert status["turn"] == "black"

    async def test_no_active_game_errors(self, server: MockChessServer) -> None:
        """Query tools fail with no_active_game when no game exists."""
        client = await server.create_session()
        with pytest.raises(McpError, match="no_active_game"):
            await client.get_board()
        with pytest.raises(McpError, match="no_active_game"):
            await client.get_status()
        with pytest.raises(McpError, match="no_active_game"):
            await client.get_legal_moves()
        with pytest.raises(McpError, match="no_active_game"):
            await client.get_history()

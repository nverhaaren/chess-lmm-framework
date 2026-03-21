"""In-memory mock chess server using python-chess for rules.

Implements ChessSessionClient and ChessServerFactory protocols
without inheriting from them (structural typing via Protocol).

Out of scope: clock/timed games, PGN export, spectator mode, messages.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

import chess

from chess_lmm.types import (
    BoardResult,
    CanClaimDraw,
    Color,
    CreateGameResult,
    DeclineDrawResult,
    DoneResult,
    ExportResult,
    GameStatus,
    GameStatusValue,
    HistoryEntry,
    HistoryMoveHalf,
    HistoryResult,
    JoinGameResult,
    LegalMovesResult,
    MakeMoveResult,
    McpError,
    MessagesResult,
    MoveNotation,
    OfferDrawResult,
    ResultValue,
    SendMessageResult,
    ServerState,
)

# Map python-chess outcomes to our types
_PIECE_SYMBOL_UPPER = {
    chess.PAWN: "",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def _move_to_lan(board: chess.Board, move: chess.Move) -> str:
    """Convert a python-chess Move to LAN notation (e.g. 'e2e4', 'e7e8q')."""
    lan = move.uci()
    return lan


def _move_to_san(board: chess.Board, move: chess.Move) -> str:
    """Convert a python-chess Move to SAN notation.

    Uses python-chess's built-in SAN generation.
    """
    return board.san(move)


def _derive_server_state(game_status: GameStatusValue) -> ServerState:
    """Derive server_state from game_status per spec Section 8.5.2."""
    if game_status == "no_game":
        return "no_game"
    if game_status == "awaiting_players":
        return "awaiting_players"
    if game_status in ("ongoing", "check"):
        return "ongoing"
    return "game_over"


def _color_name(color: chess.Color) -> Color:
    """Convert python-chess color bool to string."""
    return "white" if color == chess.WHITE else "black"


class _HistoryRecord:
    """Tracks a single half-move in the game history."""

    def __init__(self, san: str, lan: str, color: chess.Color) -> None:
        self.san = san
        self.lan = lan
        self.color = color


class MockChessGame:
    """Shared game state for all sessions connected to this mock server.

    State machine: no_game -> awaiting_players -> ongoing -> game_over
    """

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.game_id: str | None = None
        self.initial_fullmove: int = 1
        self._state: GameStatusValue = "no_game"
        self._players: dict[Color, str] = {}  # color -> session_id
        self._session_colors: dict[str, Color] = {}  # session_id -> color
        self._history: list[_HistoryRecord] = []
        self._draw_offers: dict[Color, bool] = {"white": False, "black": False}
        self._result: ResultValue | None = None
        self._termination_reason: str | None = None
        self._done_sessions: set[str] = set()

    @property
    def server_state(self) -> ServerState:
        return _derive_server_state(self._state)

    @property
    def game_status(self) -> GameStatusValue:
        return self._state

    def _require_state(self, *allowed: ServerState) -> None:
        """Raise McpError if not in one of the allowed server states."""
        if self.server_state not in allowed:
            if self.server_state == "no_game":
                raise McpError("no_active_game", "No game exists.")
            if self.server_state == "game_over":
                raise McpError("game_over", "The game has already ended.")
            raise McpError(
                "wrong_state",
                f"Server is in '{self.server_state}' state.",
            )

    def _require_joined(self, session_id: str) -> Color:
        """Raise McpError if session hasn't joined; return color."""
        if session_id not in self._session_colors:
            raise McpError("not_joined", "Session has not joined the game.")
        return self._session_colors[session_id]

    def _require_player(self, session_id: str) -> Color:
        """Require joined as a player (not spectator). Return color."""
        color = self._require_joined(session_id)
        return color

    def _require_turn(self, session_id: str) -> Color:
        """Require it's this player's turn. Return color."""
        color = self._require_player(session_id)
        assert self.board is not None
        current_turn = _color_name(self.board.turn)
        if color != current_turn:
            raise McpError(
                "not_your_turn",
                f"It is {current_turn.capitalize()}'s turn.",
            )
        return color

    def _opponent_color(self, color: Color) -> Color:
        return "black" if color == "white" else "white"

    def _update_game_status(self) -> None:
        """Update game status after a move, checking for terminal conditions."""
        assert self.board is not None

        # Check automatic draws first (75-move, fivefold repetition)
        # but checkmate takes priority per spec Section 4.5
        outcome = self.board.outcome(claim_draw=False)

        if outcome is not None:
            if outcome.termination == chess.Termination.CHECKMATE:
                self._state = "checkmate"
                winner = _color_name(not self.board.turn)
                self._result = "1-0" if winner == "white" else "0-1"
                self._termination_reason = (
                    f"Checkmate \u2014 {winner.capitalize()} wins"
                )
                return
            if outcome.termination == chess.Termination.STALEMATE:
                self._state = "stalemate"
                self._result = "1/2-1/2"
                self._termination_reason = "Stalemate \u2014 draw"
                return
            if outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                self._state = "draw_automatic_fifty"
                self._result = "1/2-1/2"
                self._termination_reason = (
                    "Draw by 75-move rule \u2014 automatic under FIDE rules"
                )
                return
            if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                self._state = "draw_automatic_repetition"
                self._result = "1/2-1/2"
                self._termination_reason = (
                    "Draw by fivefold repetition \u2014 automatic under FIDE rules"
                )
                return

        # Still ongoing - check if in check
        if self.board.is_check():
            self._state = "check"
        else:
            self._state = "ongoing"

    def _build_status(self, session_id: str) -> GameStatus:
        """Build a full GameStatus response."""
        if self.board is None:
            return GameStatus(
                game_id=None,
                server_state="no_game",
                game_status="no_game",
                turn=None,
                fen=None,
                fullmove_number=0,
                halfmove_clock=0,
                is_check=False,
                can_claim_draw=CanClaimDraw(fifty_move=False, repetition=False),
                insufficient_material=False,
                draw_offered=False,
                last_move=None,
                result=None,
                termination_reason=None,
                clock=None,
            )

        turn: Color | None = None
        if self.server_state == "ongoing":
            turn = _color_name(self.board.turn)

        # Last move
        last_move: MoveNotation | None = None
        if self._history:
            last = self._history[-1]
            last_move = MoveNotation(san=last.san, lan=last.lan)

        # Draw offered to this session?
        draw_offered = False
        if session_id in self._session_colors:
            my_color = self._session_colors[session_id]
            opponent = self._opponent_color(my_color)
            draw_offered = self._draw_offers.get(opponent, False)

        # Can claim draw?
        can_fifty = self.board.can_claim_fifty_moves()
        can_rep = self.board.can_claim_threefold_repetition()

        return GameStatus(
            game_id=self.game_id,
            server_state=self.server_state,
            game_status=self.game_status,
            turn=turn,
            fen=self.board.fen(),
            fullmove_number=self.board.fullmove_number,
            halfmove_clock=self.board.halfmove_clock,
            is_check=self.board.is_check(),
            can_claim_draw=CanClaimDraw(fifty_move=can_fifty, repetition=can_rep),
            insufficient_material=self.board.is_insufficient_material(),
            draw_offered=draw_offered,
            last_move=last_move,
            result=self._result,
            termination_reason=self._termination_reason,
            clock=None,
        )

    # --- Game management tools ---

    def create_game(
        self,
        *,
        fen: str | None = None,
        history: list[str] | None = None,
    ) -> CreateGameResult:
        self._require_state("no_game")

        # Set up board
        if fen is not None:
            try:
                board = chess.Board(fen)
            except ValueError as e:
                raise McpError("invalid_fen", str(e)) from e
            # Validate position
            status = board.status()
            if status != chess.STATUS_VALID:
                raise McpError(
                    "invalid_fen",
                    f"Invalid position: {_describe_board_status(status)}",
                )
        else:
            board = chess.Board()

        self.initial_fullmove = board.fullmove_number

        # Replay history if provided
        if history:
            for i, move_str in enumerate(history):
                parsed = _parse_move(board, move_str)
                if parsed is None:
                    raise McpError(
                        "invalid_history",
                        f"Move {i + 1} ('{move_str}') is illegal in position.",
                    )
                board.push(parsed)

        self.board = board
        self.game_id = str(uuid.uuid4())
        self._state = "awaiting_players"

        return CreateGameResult(
            game_id=self.game_id,
            fen=self.board.fen(),
            game_status="awaiting_players",
            time_control=None,
        )

    def join_game(
        self,
        session_id: str,
        color: str,
        *,
        name: str | None = None,
    ) -> JoinGameResult:
        self._require_state("awaiting_players", "ongoing")

        if session_id in self._session_colors:
            raise McpError("already_joined", "This session has already joined.")

        if color == "spectator":
            raise McpError(
                "spectator_not_allowed",
                "Spectator mode not supported in mock server.",
            )

        if color == "random":
            # Assign to first open slot
            if "white" not in self._players:
                color = "white"
            elif "black" not in self._players:
                color = "black"
            else:
                raise McpError("color_taken", "Both colors are already claimed.")
        else:
            if color not in ("white", "black"):
                raise McpError(
                    "color_taken",
                    f"Invalid color: '{color}'. Must be 'white', 'black', or 'random'.",
                )

        assert color in ("white", "black")
        typed_color: Color = "white" if color == "white" else "black"

        if typed_color in self._players:
            raise McpError(
                "color_taken",
                f"{typed_color.capitalize()} is already claimed.",
            )

        self._players[typed_color] = session_id
        self._session_colors[session_id] = typed_color

        # Transition to ongoing when both players joined
        if len(self._players) == 2:
            self._state = "ongoing"
            # Check for immediate terminal state or check
            self._update_game_status()

        # For the join response, map to the two valid values
        join_status: Literal["awaiting_players", "ongoing"] = (
            "awaiting_players" if self.game_status == "awaiting_players" else "ongoing"
        )

        return JoinGameResult(
            assigned_color=typed_color,
            game_status=join_status,
        )

    def done(self, session_id: str) -> DoneResult:
        if self.server_state != "game_over":
            raise McpError("game_not_over", "The game is still in progress.")
        if session_id in self._done_sessions:
            raise McpError("already_done", "This client has already sent done.")
        self._done_sessions.add(session_id)
        remaining = len(self._players) - len(self._done_sessions)
        return DoneResult(acknowledged=True, clients_remaining=max(0, remaining))

    def export_game(
        self,
        session_id: str,
        *,
        format: str = "pgn",
    ) -> ExportResult:
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")

        if format == "fen":
            assert self.board is not None
            return ExportResult(content=self.board.fen())

        # PGN export not implemented in mock
        raise McpError(
            "internal_error",
            "PGN export not implemented in mock server.",
            detail="mock_server limitation",
        )

    # --- Query tools ---

    def get_board(self, session_id: str) -> BoardResult:
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")
        self._require_joined(session_id)
        assert self.board is not None
        return BoardResult(fen=self.board.fen())

    def get_status(self, session_id: str) -> GameStatus:
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")
        self._require_joined(session_id)
        return self._build_status(session_id)

    def get_legal_moves(
        self,
        session_id: str,
        *,
        square: str | None = None,
        format: str = "both",
    ) -> LegalMovesResult:
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")
        self._require_joined(session_id)
        assert self.board is not None

        moves = list(self.board.legal_moves)

        # Filter by square if provided
        if square is not None:
            try:
                sq = chess.parse_square(square.lower())
            except ValueError as e:
                raise McpError(
                    "invalid_square",
                    f"'{square}' is not a valid square name.",
                ) from e
            moves = [m for m in moves if m.from_square == sq]

        # Sort by LAN
        moves.sort(key=lambda m: m.uci())

        # Build response
        result_moves: list[MoveNotation] = []
        for m in moves:
            entry: MoveNotation = {}
            if format in ("san", "both"):
                entry["san"] = _move_to_san(self.board, m)
            if format in ("lan", "both"):
                entry["lan"] = _move_to_lan(self.board, m)
            result_moves.append(entry)

        return LegalMovesResult(moves=result_moves, count=len(result_moves))

    def get_history(
        self,
        session_id: str,
        *,
        format: str = "san",
    ) -> HistoryResult:
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")
        self._require_joined(session_id)

        entries: list[HistoryEntry] = []
        move_number = self.initial_fullmove

        i = 0
        while i < len(self._history):
            rec = self._history[i]

            white_half: HistoryMoveHalf | None = None
            black_half: HistoryMoveHalf | None = None

            if rec.color == chess.WHITE:
                white_half = _build_history_half(rec, format)
                i += 1
                if i < len(self._history) and self._history[i].color == chess.BLACK:
                    black_half = _build_history_half(self._history[i], format)
                    i += 1
            else:
                # Game started with Black to move (custom FEN)
                black_half = _build_history_half(rec, format)
                i += 1

            entries.append(
                HistoryEntry(
                    move_number=move_number,
                    white=white_half,
                    black=black_half,
                )
            )
            move_number += 1

        return HistoryResult(
            moves=entries,
            total_half_moves=len(self._history),
        )

    def get_messages(
        self,
        session_id: str,
        *,
        clear: bool = True,
    ) -> MessagesResult:
        """Messages not implemented in mock — returns empty list."""
        if self.server_state == "no_game":
            raise McpError("no_active_game", "No game exists.")
        self._require_joined(session_id)
        return MessagesResult(messages=[])

    # --- Action tools ---

    def make_move(self, session_id: str, move: str) -> MakeMoveResult:
        self._require_state("ongoing")
        color = self._require_turn(session_id)
        assert self.board is not None

        # Check pending draw offer
        opponent = self._opponent_color(color)
        if self._draw_offers.get(opponent, False):
            raise McpError(
                "pending_draw_offer",
                "Opponent has a pending draw offer; must accept or decline first.",
            )

        # Parse move
        try:
            parsed = _parse_move(self.board, move)
        except _AmbiguousMoveError as e:
            raise McpError(
                "ambiguous_move",
                f"Ambiguous move: '{move}' matches multiple legal moves.",
            ) from e
        if parsed is None:
            # Distinguish between invalid format and illegal move.
            # If the string looks like a valid UCI coordinate pair or a
            # recognizable SAN piece move, it's "illegal_move"; otherwise
            # "invalid_format".
            if _looks_like_move(move):
                raise McpError(
                    "illegal_move",
                    f"Illegal move: '{move}' is not legal in this position.",
                )
            raise McpError(
                "invalid_format",
                f"Could not parse '{move}' as SAN or LAN.",
            )

        # Record before pushing (need SAN from current position)
        san = _move_to_san(self.board, parsed)
        lan = _move_to_lan(self.board, parsed)

        # Push the move
        self.board.push(parsed)
        self._history.append(
            _HistoryRecord(
                san=san, lan=lan, color=chess.WHITE if color == "white" else chess.BLACK
            )
        )

        # Withdraw any draw offer from the moving player
        self._draw_offers[color] = False

        # Update game status
        self._update_game_status()

        status = self._build_status(session_id)
        return MakeMoveResult(
            move_played=MoveNotation(san=san, lan=lan),
            **status,
        )

    def claim_draw(self, session_id: str) -> GameStatus:
        self._require_state("ongoing")
        color = self._require_turn(session_id)
        assert self.board is not None

        # Check fifty-move first (priority per spec)
        if self.board.can_claim_fifty_moves():
            self._state = "draw_fifty_move"
            self._result = "1/2-1/2"
            claimant = color.capitalize()
            self._termination_reason = (
                f"Draw by fifty-move rule (claimed by {claimant})"
            )
            return self._build_status(session_id)

        if self.board.can_claim_threefold_repetition():
            self._state = "draw_repetition"
            self._result = "1/2-1/2"
            claimant = color.capitalize()
            self._termination_reason = (
                f"Draw by threefold repetition (claimed by {claimant})"
            )
            return self._build_status(session_id)

        raise McpError(
            "draw_not_available",
            "Neither fifty-move nor threefold repetition conditions are met.",
        )

    def offer_draw(self, session_id: str) -> OfferDrawResult:
        self._require_state("ongoing")
        color = self._require_player(session_id)

        if self._draw_offers[color]:
            raise McpError(
                "already_offered",
                "A draw offer from this player is already pending.",
            )

        self._draw_offers[color] = True
        return OfferDrawResult(offered=True)

    def accept_draw(self, session_id: str) -> GameStatus:
        self._require_state("ongoing")
        color = self._require_player(session_id)
        opponent = self._opponent_color(color)

        if not self._draw_offers[opponent]:
            raise McpError("no_pending_offer", "No draw offer to accept.")

        self._state = "draw_agreement"
        self._result = "1/2-1/2"
        self._termination_reason = "Draw by agreement"
        self._draw_offers = {"white": False, "black": False}

        return self._build_status(session_id)

    def decline_draw(self, session_id: str) -> DeclineDrawResult:
        self._require_state("ongoing")
        color = self._require_player(session_id)
        opponent = self._opponent_color(color)

        if not self._draw_offers[opponent]:
            raise McpError("no_pending_offer", "No draw offer to decline.")

        self._draw_offers[opponent] = False
        return DeclineDrawResult(declined=True)

    def resign(self, session_id: str) -> GameStatus:
        self._require_state("ongoing")
        color = self._require_player(session_id)
        winner = self._opponent_color(color)

        self._state = "resigned"
        self._result = "1-0" if winner == "white" else "0-1"
        self._termination_reason = (
            f"{color.capitalize()} resigns \u2014 {winner.capitalize()} wins"
        )

        return self._build_status(session_id)

    def send_message(self, session_id: str, text: str) -> SendMessageResult:
        """Messages not implemented in mock."""
        self._require_state("ongoing")
        self._require_player(session_id)
        return SendMessageResult(sent=True)


class MockSessionClient:
    """Per-session client wrapping a shared MockChessGame.

    Satisfies the ChessSessionClient protocol.
    """

    def __init__(self, game: MockChessGame, session_id: str) -> None:
        self._game = game
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    # --- Game management ---

    async def create_game(
        self,
        *,
        fen: str | None = None,
        history: list[str] | None = None,
    ) -> CreateGameResult:
        return self._game.create_game(fen=fen, history=history)

    async def join_game(
        self,
        color: str,
        *,
        name: str | None = None,
    ) -> JoinGameResult:
        return self._game.join_game(self._session_id, color, name=name)

    async def export_game(
        self,
        *,
        format: str = "pgn",
    ) -> ExportResult:
        return self._game.export_game(self._session_id, format=format)

    async def done(self) -> DoneResult:
        return self._game.done(self._session_id)

    # --- Query tools ---

    async def get_board(self) -> BoardResult:
        return self._game.get_board(self._session_id)

    async def get_status(self) -> GameStatus:
        return self._game.get_status(self._session_id)

    async def get_legal_moves(
        self,
        *,
        square: str | None = None,
        format: str = "both",
    ) -> LegalMovesResult:
        return self._game.get_legal_moves(
            self._session_id, square=square, format=format
        )

    async def get_history(
        self,
        *,
        format: str = "san",
    ) -> HistoryResult:
        return self._game.get_history(self._session_id, format=format)

    async def get_messages(
        self,
        *,
        clear: bool = True,
    ) -> MessagesResult:
        return self._game.get_messages(self._session_id, clear=clear)

    # --- Action tools ---

    async def make_move(self, move: str) -> MakeMoveResult:
        return self._game.make_move(self._session_id, move)

    async def claim_draw(self) -> GameStatus:
        return self._game.claim_draw(self._session_id)

    async def offer_draw(self) -> OfferDrawResult:
        return self._game.offer_draw(self._session_id)

    async def accept_draw(self) -> GameStatus:
        return self._game.accept_draw(self._session_id)

    async def decline_draw(self) -> DeclineDrawResult:
        return self._game.decline_draw(self._session_id)

    async def resign(self) -> GameStatus:
        return self._game.resign(self._session_id)

    async def send_message(self, text: str) -> SendMessageResult:
        return self._game.send_message(self._session_id, text)


class MockChessServer:
    """Factory that creates per-session clients for the mock server.

    Satisfies the ChessServerFactory protocol.
    """

    def __init__(self) -> None:
        self._game = MockChessGame()
        self._session_counter = 0

    @property
    def game(self) -> MockChessGame:
        """Access the underlying game state (for testing)."""
        return self._game

    def create_session(self) -> MockSessionClient:
        """Create a new session client with a unique session ID."""
        self._session_counter += 1
        session_id = f"session-{self._session_counter}"
        return MockSessionClient(self._game, session_id)


# --- Helper functions ---


class _AmbiguousMoveError(Exception):
    """Raised when SAN input matches multiple legal moves."""


def _normalize_san(move_str: str) -> str:
    """Apply SAN normalization per spec Section 6.3.

    1. Strip NAGs ($N).
    2. Strip move assessment glyphs (longest first).
    3. Convert digit-zero castling to letter-O castling.
    4. Strip leading/trailing whitespace.
    """
    s = move_str.strip()
    # 1. Strip NAGs
    s = re.sub(r"\$\d+", "", s).strip()
    # 2. Strip assessment glyphs (longest first)
    for glyph in ("!!", "??", "!?", "?!", "!", "?"):
        if s.endswith(glyph):
            s = s[: -len(glyph)]
            break
    # 3. Convert digit-zero castling
    if s == "0-0-0":
        s = "O-O-O"
    elif s == "0-0":
        s = "O-O"
    return s.strip()


def _parse_move(board: chess.Board, move_str: str) -> chess.Move | None:
    """Parse a move string per spec Section 8.4.

    1. Apply SAN normalization (Section 6.3).
    2. Try SAN parse.
    3. LAN fallback on original string.

    Returns the parsed Move or None if both fail.
    Raises _AmbiguousMoveError if SAN is ambiguous.
    """
    # 1. Normalize
    normalized = _normalize_san(move_str)

    # 2. Try SAN on normalized string
    try:
        return board.parse_san(normalized)
    except chess.AmbiguousMoveError as e:
        raise _AmbiguousMoveError(normalized) from e
    except (chess.IllegalMoveError, chess.InvalidMoveError):
        pass

    # 3. LAN fallback on original string (not normalized)
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass

    return None


def _build_history_half(rec: _HistoryRecord, format: str) -> HistoryMoveHalf:
    """Build a history half-move entry."""
    half = HistoryMoveHalf(clock_ms=None)
    if format in ("san", "both"):
        half["san"] = rec.san
    if format in ("lan", "both"):
        half["lan"] = rec.lan
    return half


_UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
_SAN_PIECE_PATTERN = re.compile(r"^[KQRBN][a-h]?[1-8]?x?[a-h][1-8]")
_SAN_PAWN_PATTERN = re.compile(r"^[a-h](x[a-h])?[1-8]")
_CASTLING_PATTERN = re.compile(r"^O-O(-O)?$|^0-0(-0)?$")


def _looks_like_move(s: str) -> bool:
    """Check if a string looks like a plausible chess move format.

    Used to distinguish 'illegal_move' from 'invalid_format' errors.
    """
    s = s.strip().rstrip("+#!?")
    if _UCI_PATTERN.match(s):
        return True
    if _SAN_PIECE_PATTERN.match(s):
        return True
    if _SAN_PAWN_PATTERN.match(s):
        return True
    return bool(_CASTLING_PATTERN.match(s))


def _describe_board_status(status: chess.Status) -> str:
    """Describe a python-chess board status bitmask as human-readable string."""
    descriptions: list[str] = []
    if status & chess.STATUS_NO_WHITE_KING:
        descriptions.append("no white king")
    if status & chess.STATUS_NO_BLACK_KING:
        descriptions.append("no black king")
    if status & chess.STATUS_TOO_MANY_KINGS:
        descriptions.append("too many kings")
    if status & chess.STATUS_TOO_MANY_WHITE_PIECES:
        descriptions.append("too many white pieces")
    if status & chess.STATUS_TOO_MANY_BLACK_PIECES:
        descriptions.append("too many black pieces")
    if status & chess.STATUS_PAWNS_ON_BACKRANK:
        descriptions.append("pawns on back rank")
    if status & chess.STATUS_TOO_MANY_WHITE_PAWNS:
        descriptions.append("too many white pawns")
    if status & chess.STATUS_TOO_MANY_BLACK_PAWNS:
        descriptions.append("too many black pawns")
    if status & chess.STATUS_BAD_CASTLING_RIGHTS:
        descriptions.append("bad castling rights")
    if status & chess.STATUS_INVALID_EP_SQUARE:
        descriptions.append("invalid en passant square")
    if status & chess.STATUS_OPPOSITE_CHECK:
        descriptions.append("opponent in check")
    if not descriptions:
        descriptions.append("unknown status issue")
    return ", ".join(descriptions)

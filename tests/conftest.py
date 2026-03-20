"""Shared test fixtures for chess_lmm tests."""

from __future__ import annotations

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Scholar's mate position (after 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#)
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"

# Stalemate position (lone king, Section 9.2.1)
STALEMATE_FEN = "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"

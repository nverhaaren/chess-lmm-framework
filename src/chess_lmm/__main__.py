"""Entry point for `python -m chess_lmm`."""

import sys


def main() -> None:
    """Launch the chess LMM orchestrator."""
    # Orchestrator implementation deferred to PR 4
    print("chess-lmm: orchestrator not yet implemented", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

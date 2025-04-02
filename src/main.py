import argparse
from core.game_registry import get_game_main


def main():
    """
    Command-line interface for launching a specific LinkedIn puzzle agent.

    Usage:
        python entrypoint.py --game queens --username USER --password PASS

    Supported games are registered in `core/game_registry.py`.
    """
    parser = argparse.ArgumentParser(description="Run a LinkedIn puzzle game agent.")
    parser.add_argument('--game', required=True, help="Name of the game (e.g., queens, tango, zip)")
    parser.add_argument('--username', required=True, help="LinkedIn username (email)")
    parser.add_argument('--password', required=True, help="LinkedIn password")
    args = parser.parse_args()

    game_runner = get_game_main(args.game)
    if game_runner:
        game_runner(args.username, args.password)
    else:
        print(f"[✗] Unsupported game: '{args.game}'")


if __name__ == "__main__":
    main()
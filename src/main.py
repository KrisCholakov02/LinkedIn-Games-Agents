import argparse
from core.game_registry import get_game_main


def main():
    parser = argparse.ArgumentParser(description="Run a LinkedIn puzzle game agent.")
    parser.add_argument('--game', required=True, help="Game to play (e.g., queens)")
    parser.add_argument('--username', required=True, help="LinkedIn username")
    parser.add_argument('--password', required=True, help="LinkedIn password")
    args = parser.parse_args()

    game_runner = get_game_main(args.game)
    if game_runner:
        game_runner(args.username, args.password)
    else:
        print(f"[✗] Game '{args.game}' is not supported.")

if __name__ == "__main__":
    main()
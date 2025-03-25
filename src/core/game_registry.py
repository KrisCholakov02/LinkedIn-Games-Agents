from src.games.queens.main import run_game as run_queens

def get_game_main(name: str):
    return {
        "queens": run_queens,
        # "tange": run_tange (future)
    }.get(name.lower())
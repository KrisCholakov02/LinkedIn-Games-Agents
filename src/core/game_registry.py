from src.games.queens.main import run_game as run_queens
from src.games.tango.main import run_game as run_tango

def get_game_main(name: str):
    return {
        "queens": run_queens,
        "tango": run_tango
    }.get(name.lower())
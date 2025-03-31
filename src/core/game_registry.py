from src.games.queens.main import run_game as run_queens
from src.games.tango.main import run_game as run_tango
from src.games.zip.main import run_game as run_zip

def get_game_main(name: str):
    return {
        "queens": run_queens,
        "tango": run_tango,
        "zip": run_zip
    }.get(name.lower())
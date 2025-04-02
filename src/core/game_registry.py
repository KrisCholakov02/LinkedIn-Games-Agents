from src.games.queens.main import run_game as run_queens
from src.games.tango.main import run_game as run_tango
from src.games.zip.main import run_game as run_zip


def get_game_main(name: str):
    """
    Returns the main execution function for a given game name.

    Args:
        name (str): Name of the game ('queens', 'tango', or 'zip').

    Returns:
        Callable or None: The corresponding game's run_game function,
        or None if the name is invalid.
    """
    return {
        "queens": run_queens,
        "tango": run_tango,
        "zip": run_zip
    }.get(name.lower())
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.2.3",
#     "ryd-numerov >= 0.5.1",
# ]
# ///

__version__ = "1.1"

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the generate_database script."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  uv run generate_database.py Rb --log-level INFO\n"),
    )
    parser.add_argument("species", help="The species to generate the database for.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level (default: INFO)",
    )

    args = parser.parse_args()
    configure_logging(args.log_level, args.species)
    create_database_one_species(args.species)


def configure_logging(log_level: str, species: str) -> None:
    """Initialize the logger."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    stream_formatter = logging.Formatter("%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    file_formatter = logging.Formatter("%(levelname)s: %(message)s")
    log_file = Path.cwd() / "database" / f"{species}_v{__version__}" / "log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def create_database_one_species(species: str) -> None:
    """Create database for a given species."""
    logger.info("Start creating database for: %s and version: v%s", species, __version__)


if __name__ == "__main__":
    main()

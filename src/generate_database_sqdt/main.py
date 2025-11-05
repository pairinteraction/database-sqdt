from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import ryd_numerov
from ryd_numerov import RydbergStateAlkali
from ryd_numerov.species import SpeciesObject

from generate_database_sqdt import __version__, database_sql_file
from generate_database_sqdt.generate_misc import create_tables_for_misc
from generate_database_sqdt.utils import (
    calc_matrix_element_one_pair,
    get_radial_state_cached,
    get_sorted_list_of_states,
)

if TYPE_CHECKING:
    from ryd_numerov.units import MatrixElementOperator


class WarningsAsExceptionsHandler(logging.Handler):
    """Custom logging handler to raise exceptions for errors."""

    def __init__(self) -> None:
        super().__init__()
        self.warnings_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            raise RuntimeError(record.getMessage())


logger = logging.getLogger(__name__)


MATRIX_ELEMENTS_OF_INTEREST: dict[str, MatrixElementOperator] = {
    # "key": "operator"
    "matrix_elements_d": "electric_dipole",
    "matrix_elements_q": "electric_quadrupole",
    "matrix_elements_o": "electric_octupole",
    "matrix_elements_q0": "electric_quadrupole_zero",
    "matrix_elements_mu": "magnetic_dipole",
}


def main() -> None:
    """Entry point for the generate_database script."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  uv run generate_database.py Rb --log-level INFO\n"),
    )
    parser.add_argument("species", help="The species name to generate the database for.")
    parser.add_argument(
        "--n-min",
        default=1,
        type=int,
        help="The minimal principal quantum number n for the states to be included in the database. "
        "This is used for species, where the low lying states do not converge nicely, so we exclude those states. "
        "Default 1 will start with the ground state configuration of the specific species (e.g. n_min=5 for Rb).",
    )
    parser.add_argument(
        "--n-max",
        default=120,
        type=int,
        help="The maximum principal quantum number n for the states to be included in the database.",
    )
    parser.add_argument(
        "--directory",
        default="database",
        type=str,
        help="The directory where the database will be saved.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--warnings-as-exceptions",
        action="store_true",
        help="Treat warnings in ryd_numerov as exceptions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the species folder if it exists and create a new one.",
    )

    args = parser.parse_args()
    species_folder = Path(args.directory) / f"{args.species}_v{__version__}"
    if args.species == "misc":
        species_folder = Path(args.directory) / f"misc_v{__version__}"

    if species_folder.exists():
        if args.overwrite:
            shutil.rmtree(species_folder)
        else:
            raise FileExistsError(f"The folder {species_folder} already exists. Use --overwrite to overwrite it.")
    species_folder.mkdir(parents=True, exist_ok=True)
    os.chdir(species_folder)

    configure_logging(
        args.log_level,
        args.species,
        warnings_as_exceptions=args.warnings_as_exceptions,
    )
    time_start = time.perf_counter()
    if args.species == "misc":
        create_tables_for_misc(f_max=args.n_max, kappa_max=3)
    else:
        create_tables_for_one_species(args.species, args.n_min, args.n_max)
    logger.info("Time taken: %.2f seconds", time.perf_counter() - time_start)


def configure_logging(log_level: str, species_name: str, *, warnings_as_exceptions: bool) -> None:
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
    log_file = Path(f"{species_name}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    if warnings_as_exceptions:
        logging.getLogger().addHandler(WarningsAsExceptionsHandler())


def create_tables_for_one_species(
    species_name: str, n_min: int, n_max: int, max_delta_n: int = 10, all_n_up_to: int = 30
) -> None:
    """Create database for a given species."""
    logger.info("Start creating database for %s and version v%s", species_name, __version__)
    logger.info("n-min=%d", n_min)
    logger.info("n-max=%d", n_max)
    logger.info("max_delta_n=%d", max_delta_n)
    logger.info("all_n_up_to=%d", all_n_up_to)
    logger.info("ryd_numerov.__version__=%s", ryd_numerov.__version__)

    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(database_sql_file.read_text(encoding="utf-8"))
        list_of_states = get_sorted_list_of_states(species_name, n_min, n_max)
        populate_states_table(list_of_states, conn)  # type: ignore [arg-type]
        populate_matrix_elements_table(list_of_states, conn, max_delta_n, all_n_up_to)  # type: ignore [arg-type]
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    species = SpeciesObject.from_name(species_name)
    with sqlite3.connect(db_file) as conn:
        for tkey in ["states", *MATRIX_ELEMENTS_OF_INTEREST.keys()]:
            parquet_file = Path(f"{tkey}.parquet")
            table = pd.read_sql_query(f"SELECT * FROM {tkey}", conn)
            if tkey == "states":
                table = table.astype({"is_j_total_momentum": bool, "is_calculated_with_mqdt": bool})
                table["is_j_total_momentum"] = species.i_c == 0 or species.i_c is None
                table["is_calculated_with_mqdt"] = False
            table.to_parquet(parquet_file, index=False, compression="zstd")
            logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
            if logging.getLogger().isEnabledFor(logging.INFO):
                table.info(verbose=True)
                with Path(f"{species_name}.log").open("a") as buf:
                    table.info(buf=buf)

    logger.info("get_radial_state_cached: %s", get_radial_state_cached.cache_info())


def populate_states_table(
    list_of_states: list[RydbergStateAlkali],  # | list[RydbergStateAlkalineLS]
    conn: sqlite3.Connection,
) -> None:
    """Populate the states table with data for a given species."""
    states_data = []
    for ids, state in enumerate(list_of_states):
        angular_ket = state.angular
        angular_state = state.angular.to_state()

        states_data.append(
            (
                ids,  # id, will be set later
                state.species.get_ionization_energy() + state.get_energy("a.u."),  # energy
                (-1) ** state.l,  # parity = (-1)^l
                state.n,  # n: quantum number
                state.get_nu(),  # nu = NStar for sqdt
                angular_ket.f_tot,  # f: quantum number
                state.get_nu(),  # exp_nui = nu for sqdt
                angular_ket.l_tot,  # exp_l = l
                angular_ket.j_tot,  # exp_j = j
                angular_ket.s_tot,  # exp_s = s
                angular_ket.l_r,  # exp_l_ryd = l for sqdt
                angular_state.calc_exp_qn("j_r"),  # exp_j_ryd = j for sqdt only one valence electron
                0,  # std_nui = 0
                0,  # std_l = 0
                0,  # std_j = 0
                0,  # std_s = 0
                0,  # std_l_ryd = 0
                angular_state.calc_std_qn("j_r"),  # std_j_ryd = 0 for sqdt and only one valence electron
                "True",  # is_j_total_momentum = True for no hyperfine splitting
                "False",  # is_calculated_with_mqdt = False for sqdt
                0,  # underspecified_channel_contribution = 0 for sqdt
            )
        )

    conn.executemany(f"INSERT INTO states VALUES ({', '.join(['?'] * len(states_data[0]))})", states_data)
    logger.info("Created the 'states' table (%s rows)", conn.execute("SELECT COUNT(*) FROM states").fetchone()[0])


def populate_matrix_elements_table(
    list_of_states: list[RydbergStateAlkali],  # | list[RydbergStateAlkalineLS]
    conn: sqlite3.Connection,
    max_delta_n: int,
    all_n_up_to: int,
) -> None:
    k_angular_max = 3

    list_of_id_state = [(ids, state) for ids, state in enumerate(list_of_states)]
    # sort the states by l, n for more efficient caching
    list_of_id_state = sorted(list_of_id_state, key=lambda x: (x[1].l, x[1].n, x[0]))

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i, (id1, state1) in enumerate(list_of_id_state):
        list_filtered = filter(lambda x: x[1].l - state1.l <= k_angular_max, list_of_id_state[i:])
        for id2, state2 in list_filtered:
            if all(n > all_n_up_to for n in [state1.n, state2.n]) and abs(state1.n - state2.n) > max_delta_n:
                # If delta_n is larger than max_delta_n, we dont calculate the matrix elements anymore,
                # since these are so small, that they are usually not relevant for further calculations
                # However, we keep all dipole interactions with small n (we choose all_n_up_to as a cutoff)
                # since these are relevant for the spontaneous decay rates
                continue

            id_tuple = (id1, id2) if id1 <= id2 else (id2, id1)
            states = (state1, state2) if id1 <= id2 else (state2, state1)

            me_one_pair = calc_matrix_element_one_pair(states[0], states[1], MATRIX_ELEMENTS_OF_INTEREST)
            for tkey, me in me_one_pair.items():
                matrix_elements[tkey].append((*id_tuple, me))

            if id1 != id2:
                id_tuple = (id_tuple[1], id_tuple[0])
                me_one_pair = calc_matrix_element_one_pair(states[1], states[0], MATRIX_ELEMENTS_OF_INTEREST)
                for tkey, me in me_one_pair.items():
                    matrix_elements[tkey].append((*id_tuple, me))

    for tkey, mes in matrix_elements.items():
        conn.executemany(f"INSERT INTO {tkey} VALUES (?, ?, ?)", sorted(mes))
        logger.info(
            "Created the '%s' table (%s rows)", tkey, conn.execute(f"SELECT COUNT(*) FROM {tkey}").fetchone()[0]
        )


if __name__ == "__main__":
    main()

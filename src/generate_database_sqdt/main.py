import argparse
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import ryd_numerov
from ryd_numerov.angular.utils import clebsch_gordan_6j
from ryd_numerov.elements import BaseElement
from ryd_numerov.rydberg import RydbergState

from generate_database_sqdt import __version__, database_sql_file
from generate_database_sqdt.generate_misc import create_tables_for_misc
from generate_database_sqdt.utils import (
    _calc_radial_matrix_element_cached,
    calc_matrix_element_one_pair,
    calc_reduced_angular_matrix_element_cached,
    filter_qns,
    get_rydberg_state_cached,
    get_sorted_list_of_states,
)

if TYPE_CHECKING:
    from ryd_numerov.units import OperatorType


class WarningsAsExceptionsHandler(logging.Handler):
    """Custom logging handler to raise exceptions for errors."""

    def __init__(self) -> None:
        super().__init__()
        self.warnings_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            raise RuntimeError(record.getMessage())


logger = logging.getLogger(__name__)


MATRIX_ELEMENTS_OF_INTEREST: dict[str, tuple["OperatorType", int, int]] = {
    # key: (operator, k_radial, k_angular)  # noqa: ERA001
    "matrix_elements_d": ("ELECTRIC", 1, 1),  # dipole
    "matrix_elements_q": ("ELECTRIC", 2, 2),  # quadrupole
    "matrix_elements_o": ("ELECTRIC", 3, 3),  # octopole
    "matrix_elements_q0": ("ELECTRIC", 2, 0),  # diamagnetic
    "matrix_elements_mu": ("MAGNETIC", 0, 1),  # magnetic
}


def main() -> None:
    """Entry point for the generate_database script."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  uv run generate_database.py Rb --log-level INFO\n"),
    )
    parser.add_argument("species", help="The species to generate the database for.")
    parser.add_argument(
        "--n-min",
        default=1,
        type=int,
        help="The minimal principal quantum number n for the states to be included in the database. "
        "This is used for elements, where the low lying states do not converge nicely, so we exclude those states. "
        "Default 1 will start with the ground state configuration of the specific element (e.g. n_min=5 for Rb).",
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
    if args.species == "misc":
        species_folder = Path(args.directory) / f"misc_v{__version__}"
    else:
        element = BaseElement.from_species(args.species)
        suffix = "_sqdt" if element.number_valence_electrons == 2 else ""  # noqa: PLR2004 # 2 == alkaline earth atom
        species_folder = Path(args.directory) / f"{args.species}{suffix}_v{__version__}"

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


def configure_logging(log_level: str, species: str, *, warnings_as_exceptions: bool) -> None:
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
    log_file = Path(f"{species}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    if warnings_as_exceptions:
        logging.getLogger().addHandler(WarningsAsExceptionsHandler())


def create_tables_for_one_species(
    species: str, n_min: int, n_max: int, max_delta_n: int = 10, all_n_up_to: int = 30
) -> None:
    """Create database for a given species."""
    logger.info("Start creating database for %s and version v%s", species, __version__)
    logger.info("n-min=%d", n_min)
    logger.info("n-max=%d", n_max)
    logger.info("max_delta_n=%d", max_delta_n)
    logger.info("all_n_up_to=%d", all_n_up_to)
    logger.info("ryd_numerov.__version__=%s", ryd_numerov.__version__)

    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(database_sql_file.read_text(encoding="utf-8"))
        list_of_states = get_sorted_list_of_states(species, n_min, n_max)
        populate_states_table(list_of_states, conn)
        populate_matrix_elements_table(list_of_states, conn, max_delta_n, all_n_up_to)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    with sqlite3.connect(db_file) as conn:
        for tkey in ["states", *MATRIX_ELEMENTS_OF_INTEREST.keys()]:
            parquet_file = Path(f"{tkey}.parquet")
            table = pd.read_sql_query(f"SELECT * FROM {tkey}", conn)
            if tkey == "states":
                table = table.astype({"is_j_total_momentum": bool, "is_calculated_with_mqdt": bool})
                table["is_j_total_momentum"] = True
                table["is_calculated_with_mqdt"] = False
            table.to_parquet(parquet_file, index=False, compression="zstd")
            logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
            if logging.getLogger().isEnabledFor(logging.INFO):
                table.info(verbose=True)
                with Path(f"{species}.log").open("a") as buf:
                    table.info(buf=buf)

    logger.info(
        "calc_reduced_angular_matrix_element_cached: %s", calc_reduced_angular_matrix_element_cached.cache_info()
    )
    logger.info("_calc_radial_matrix_element_cached: %s", _calc_radial_matrix_element_cached.cache_info())
    logger.info("get_rydberg_state_cached: %s", get_rydberg_state_cached.cache_info())


def populate_states_table(list_of_states: list[RydbergState], conn: "sqlite3.Connection") -> None:
    """Populate the states table with data for a given species."""
    states_data = []
    for ids, state in enumerate(list_of_states):
        std_j_ryd: float
        exp_j_ryd: float
        if state.element.number_valence_electrons == 1:
            exp_j_ryd = state.j_tot
            std_j_ryd = 0
        else:  # number_valence_electrons == 2
            s1 = s2 = 0.5
            coefficients: dict[float, float] = {
                float(j1): clebsch_gordan_6j(s1, s2, int(state.s_tot), state.l, float(j1), int(state.j_tot))
                for j1 in np.arange(abs(state.l - s1), state.l + s1 + 1)
            }
            exp_j_ryd = sum(j1 * coeff**2 for j1, coeff in coefficients.items())
            std_j_ryd_squared = sum((j1 - exp_j_ryd) ** 2 * coeff**2 for j1, coeff in coefficients.items())
            std_j_ryd = 0 if std_j_ryd_squared < 1e-12 else np.sqrt(std_j_ryd_squared)  # noqa: PLR2004

        n_star = state.get_n_star()

        states_data.append(
            (
                ids,  # id, will be set later
                state.element.get_ionization_energy() + state.get_energy("a.u."),  # energy
                (-1) ** state.l,  # parity = (-1)^l
                state.n,  # n: quantum number
                n_star,  # nu = NStar for sqdt
                state.j_tot,  # f: quantum number, neglect hyperfine splitting -> f = j_tot
                n_star,  # exp_nui = nu for sqdt
                state.l,  # exp_l = l
                state.j_tot,  # exp_j = j_tot
                state.s_tot,  # exp_s = s_tot
                state.l,  # exp_l_ryd = l for sqdt
                exp_j_ryd,  # exp_j_ryd = j_tot for sqdt only one valence electron
                0,  # std_nui = 0
                0,  # std_l = 0
                0,  # std_j = 0
                0,  # std_s = 0
                0,  # std_l_ryd = 0
                std_j_ryd,  # std_j_ryd = 0 for sqdt and only one valence electron
                "True",  # is_j_total_momentum = True for no hyperfine splitting
                "False",  # is_calculated_with_mqdt = False for sqdt
                0,  # underspecified_channel_contribution = 0 for sqdt
            )
        )

    conn.executemany(f"INSERT INTO states VALUES ({', '.join(['?'] * len(states_data[0]))})", states_data)
    logger.info("Created the 'states' table (%s rows)", conn.execute("SELECT COUNT(*) FROM states").fetchone()[0])


def populate_matrix_elements_table(
    list_of_states: list[RydbergState], conn: "sqlite3.Connection", max_delta_n: int, all_n_up_to: int
) -> None:
    k_angular_max = 3

    element = list_of_states[0].element
    list_of_qns = [(ids, state.n, state.l, state.j_tot, state.s_tot) for ids, state in enumerate(list_of_states)]

    # sort the states by s_tot, l, n for more efficient caching
    qns_sorted_by_l = np.array(sorted(list_of_qns, key=lambda x: (x[4], x[2], x[1], x[0])))

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i, (id1, n1, l1, j1, s1) in enumerate(qns_sorted_by_l):
        qns_filtered = qns_sorted_by_l[i:]
        qns_filtered = filter_qns(
            qns_filtered,
            (id1, n1, l1, j1, s1),
            all_n_up_to=all_n_up_to,
            max_delta_n=max_delta_n,
            k_angular_max=k_angular_max,
        )

        for id2, n2, l2, j2, s2 in qns_filtered:
            id_tuple = (id1, id2) if id1 <= id2 else (id2, id1)
            qns = (n1, l1, j1, s1, n2, l2, j2, s2) if id1 <= id2 else (n2, l2, j2, s2, n1, l1, j1, s1)
            me_one_pair = calc_matrix_element_one_pair(element.species, *qns, MATRIX_ELEMENTS_OF_INTEREST)
            for tkey, me in me_one_pair.items():
                matrix_elements[tkey].append((*id_tuple, me))

            if id1 != id2:
                id_tuple = (id_tuple[1], id_tuple[0])
                qns = qns[4:] + qns[:4]
                me_one_pair = calc_matrix_element_one_pair(element.species, *qns, MATRIX_ELEMENTS_OF_INTEREST)
                for tkey, me in me_one_pair.items():
                    matrix_elements[tkey].append((*id_tuple, me))

    for tkey, mes in matrix_elements.items():
        conn.executemany(f"INSERT INTO {tkey} VALUES (?, ?, ?)", sorted(mes))
        logger.info(
            "Created the '%s' table (%s rows)", tkey, conn.execute(f"SELECT COUNT(*) FROM {tkey}").fetchone()[0]
        )


if __name__ == "__main__":
    main()

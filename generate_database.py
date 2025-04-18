# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas[pyarrow]>=2.2.3",
#     "ryd-numerov >= 0.5.1",
# ]
# ///

__version__ = "1.1"

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
from ryd_numerov.angular.utils import clebsch_gordan_6j
from ryd_numerov.rydberg import RydbergState

from utils import (
    calc_matrix_element_one_pair,
    get_integrated_state,
    get_radial_matrix_element,
    get_reduced_angular_matrix_element,
    get_sorted_list_of_states,
)

if TYPE_CHECKING:
    from ryd_numerov.units import OperatorType

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
        "--overwrite",
        action="store_true",
        help="Delete the species folder if it exists and create a new one.",
    )

    args = parser.parse_args()
    species_folder = Path(args.directory) / f"{args.species}_v{__version__}"
    if args.overwrite and species_folder.exists():
        shutil.rmtree(species_folder)
    species_folder.mkdir(parents=True, exist_ok=True)
    os.chdir(species_folder)

    configure_logging(args.log_level, args.species)
    time_start = time.perf_counter()
    create_database_one_species(args.species, args.n_max)
    logger.info("Time taken: %.2f seconds", time.perf_counter() - time_start)


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
    log_file = Path(f"{species}_v{__version__}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def create_database_one_species(species: str, n_max: int) -> None:
    """Create database for a given species."""
    logger.info("Start creating database for %s and version v%s with n-max=%d", species, __version__, n_max)

    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript((Path(__file__).parent / "database.sql").read_text(encoding="utf-8"))
        list_of_states = get_sorted_list_of_states(species, n_max)
        populate_states_table(list_of_states, conn)
        populate_matrix_elements_table(list_of_states, conn)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    with sqlite3.connect(db_file) as conn:
        for tkey in ["states", *MATRIX_ELEMENTS_OF_INTEREST.keys()]:
            parquet_file = Path(f"{tkey}.parquet")
            table = pd.read_sql_query(f"SELECT * FROM {tkey}", conn)
            if tkey == "states":
                table = table.astype({"is_j_total_momentum": bool, "is_calculated_with_mqdt": bool})
            table.to_parquet(parquet_file, index=False)
            logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
            table.info(verbose=True)
            with Path(f"{species}_v{__version__}.log").open("a") as buf:
                table.info(buf=buf)

    logger.info("get_reduced_angular_matrix_element: %s", get_reduced_angular_matrix_element.cache_info())
    logger.info("get_radial_matrix_element: %s", get_radial_matrix_element.cache_info())
    logger.info("get_integrated_state: %s", get_integrated_state.cache_info())


def populate_states_table(list_of_states: list[RydbergState], conn: "sqlite3.Connection") -> None:
    """Populate the states table with data for a given species."""
    states_data = []
    for ids, state in enumerate(list_of_states):
        std_j_ryd: float
        exp_j_ryd: float
        if state.element.is_alkali:
            exp_j_ryd = state.j
            std_j_ryd = 0
        else:  # state.s in [0, 1]
            s1 = s2 = 0.5
            coefficients: dict[float, float] = {
                float(j1): clebsch_gordan_6j(s1, s2, int(state.s), state.l, float(j1), int(state.j))
                for j1 in np.arange(abs(state.l - s1), state.l + s1 + 1)
            }
            exp_j_ryd = sum(j1 * coeff**2 for j1, coeff in coefficients.items())
            std_j_ryd_squared = sum((j1 - exp_j_ryd) ** 2 * coeff**2 for j1, coeff in coefficients.items())
            std_j_ryd = 0 if std_j_ryd_squared < 1e-12 else np.sqrt(std_j_ryd_squared)  # noqa: PLR2004

        states_data.append(
            (
                ids,  # id, will be set later
                state.element.get_ionization_energy() + state.get_energy("a.u."),  # energy
                (-1) ** state.l,  # parity = (-1)^l
                state.n,  # n: quantum number
                state.quantum_defect.n_star,  # nu = NStar for sqdt
                state.j,  # f: quantum number, neglect hyperfine splitting -> f = j
                state.quantum_defect.n_star,  # exp_nui = nu for sqdt
                state.l,  # exp_l = l
                state.j,  # exp_j = j
                state.s,  # exp_s = s
                state.l,  # exp_l_ryd = l for sqdt
                exp_j_ryd,  # exp_j_ryd = j for sqdt only one valence electron
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


def populate_matrix_elements_table(list_of_states: list[RydbergState], conn: "sqlite3.Connection") -> None:
    k_angular_max = 3

    element = list_of_states[0].element
    list_of_qns = [(ids, state.n, state.l, state.j) for ids, state in enumerate(list_of_states)]

    # sort the states by l for more efficient caching
    qns_sorted_by_l = sorted(list_of_qns, key=lambda x: (x[2], x[0]))

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i, (id1, n1, l1, j1) in enumerate(qns_sorted_by_l):
        qns_filtered = filter(lambda x: x[2] - l1 <= k_angular_max, qns_sorted_by_l[i:])
        for id2, n2, l2, j2 in qns_filtered:
            # TODO add condition to break? or simply remove afterwards all elements that are to small
            # (with respect to matrix element and lifetime relevance!)

            id_tuple = (id1, id2) if id1 <= id2 else (id2, id1)
            qns = (n1, l1, j1, n2, l2, j2) if id1 <= id2 else (n2, l2, j2, n1, l1, j1)
            me_one_pair = calc_matrix_element_one_pair(element.species, *qns, MATRIX_ELEMENTS_OF_INTEREST)
            for tkey, me in me_one_pair.items():
                matrix_elements[tkey].append((*id_tuple, me))

            if id1 != id2:
                id_tuple = (id_tuple[1], id_tuple[0])
                qns = qns[3:] + qns[:3]
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

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import ryd_numerov

from generate_database_sqdt import __version__
from generate_database_sqdt.utils import calc_wigner_3j_cached

logger = logging.getLogger(__name__)


def create_tables_for_misc(f_max: float, kappa_max: int) -> None:
    """Create misc database, i.e. the wigner table."""
    logger.info("Start creating misc database for version v%s", __version__)
    logger.info("f_max=%d", f_max)
    logger.info("kappa_max=%d", kappa_max)
    logger.info("ryd_numerov.__version__=%s", ryd_numerov.__version__)

    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript((Path(__file__).parent / "database.sql").read_text(encoding="utf-8"))
        populate_wigner_table(f_max, kappa_max, conn)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    with sqlite3.connect(db_file) as conn:
        parquet_file = Path("wigner.parquet")
        table = pd.read_sql_query("SELECT * FROM wigner", conn)
        table.to_parquet(parquet_file, index=False, compression="zstd")
        logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
        table.info(verbose=True)
        with Path(f"wigner_v{__version__}.log").open("a") as buf:
            table.info(buf=buf)

    logger.info("calc_wigner_3j_cached: %s", calc_wigner_3j_cached.cache_info())


def populate_wigner_table(f_max: float, kappa_max: int, conn: "sqlite3.Connection") -> None:
    """Populate the wigner table with data for all wigner symbols up to f_max and kappa_max."""
    wigner_data = []
    for kappa in range(kappa_max + 1):
        for q in range(-kappa, kappa + 1):
            for f_initial in np.arange(0, f_max + 0.5, 0.5):
                for f_final in np.arange(np.max([f_initial % 1, f_initial - kappa]), f_initial + kappa + 1):
                    for m_initial in np.arange(-f_initial, f_initial + 1):
                        m_final = m_initial + q
                        if not -f_final <= m_final <= f_final:
                            continue
                        wigner = calc_wigner_3j_cached(f_final, kappa, f_initial, -m_final, q, m_initial)
                        wigner *= (-1) ** (f_final - m_final)
                        if wigner == 0:
                            continue
                        wigner_data.append((f_initial, f_final, m_initial, m_final, kappa, q, wigner))

    conn.executemany(f"INSERT INTO wigner VALUES ({', '.join(['?'] * len(wigner_data[0]))})", wigner_data)
    logger.info("Created the 'wigner' table (%s rows)", conn.execute("SELECT COUNT(*) FROM wigner").fetchone()[0])

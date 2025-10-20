from pathlib import Path

import pairinteraction

database_sql_file = Path(pairinteraction.__file__).parent / "_wrapped" / "database" / "database.sql"

__version__ = "1.3"

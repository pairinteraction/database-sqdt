from pathlib import Path

import pandas as pd


def main() -> None:
    # CHANGE THESE PATHS, TO THE FOLDERS YOU WANT TO COMPARE
    new_path = Path("misc_v1.2")
    old_path = Path("misc_v1.1")

    if "misc" not in new_path.name or "misc" not in old_path.name:
        raise ValueError("This script is only for comparing misc folders.")

    compare_wigner_table(new_path, old_path)


def compare_wigner_table(new_path: Path, old_path: Path) -> None:
    new = pd.read_parquet(new_path / "wigner.parquet")
    new = new.sort_values(by=["f_initial", "f_final", "kappa", "q", "m_initial", "m_final"])
    new = new.set_index(["f_initial", "f_final", "m_initial", "m_final", "kappa", "q"])

    old = pd.read_parquet(old_path / "wigner.parquet")
    old = old.sort_values(by=["f_initial", "f_final", "kappa", "q", "m_initial", "m_final"])
    old = old.set_index(["f_initial", "f_final", "m_initial", "m_final", "kappa", "q"])

    # Find entries in new that don't exist in old
    new_only = new[~new.index.isin(old.index)]
    print(f"Entries in new that don't exist in old ({len(new_only)}/{len(new)} rows):")
    if not new_only.empty:
        print(new_only)
    else:
        print("None")
    print()

    # Find entries in old that don't exist in new
    old_only = old[~old.index.isin(new.index)]
    print(f"Entries in old that don't exist in new ({len(old_only)}/{len(old)} rows):")
    if not old_only.empty:
        print(old_only)
    else:
        print("None")
    print()

    # Compare the matching entries
    new = new.drop(new_only.index)
    old = old.drop(old_only.index)

    diff = new.compare(old)
    print(f"Differences in wigner table values ({len(diff)}/{len(new)} rows):")
    print(diff)


if __name__ == "__main__":
    main()

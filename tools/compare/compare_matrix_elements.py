from pathlib import Path

import pandas as pd

TABLE_NAMES: list[str] = [
    "matrix_elements_d",
    "matrix_elements_q",
    "matrix_elements_o",
    "matrix_elements_q0",
    "matrix_elements_mu",
]


def main() -> None:
    # CHANGE THESE PATHS, TO THE FOLDERS YOU WANT TO COMPARE
    name = "Sr88_singlet"
    new_path = Path(f"{name}_v1.2")
    old_path = Path(f"{name}_v1.1")

    print(f"Comparing matrix elements tables:\n  New: {new_path}\n  Old: {old_path}")
    for table_name in TABLE_NAMES:
        if not (new_path / f"{table_name}.parquet").exists() or not (old_path / f"{table_name}.parquet").exists():
            print(f"\nSkipping {table_name} as it does not exist in either the new or the old path.")
            continue
        compare_matrix_elements_table(table_name, new_path, old_path, max_delta_n=3, min_n=16, verbose=False)


def compare_matrix_elements_table(
    table_name: str,
    new_path: Path,
    old_path: Path,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    *,
    max_delta_n: int = 3,
    min_n: int = 1,
    verbose: bool = False,
) -> None:
    """Compare the matrix elements table of two versions of the database.

    Given the path to states and matrix elements parquet tables,
    this function will
    1) create a new index mapping from a state (defined by n, exp_l, and exp_j) to a unique identifier
    2) replace the old id_initial and id_final column in the matrix elements table with this new index
    3) compare the column "val" of the two matrix elements tables.

    """
    print(f"\nComparing table    {table_name}")

    states_dict = {
        "new": pd.read_parquet(new_path / "states.parquet"),
        "old": pd.read_parquet(old_path / "states.parquet"),
    }
    table_dict = {
        "new": pd.read_parquet(new_path / f"{table_name}.parquet"),
        "old": pd.read_parquet(old_path / f"{table_name}.parquet"),
    }

    multi_index_columns = ["n", "exp_l", "exp_j"]
    if "mqdt" in str(new_path):
        multi_index_columns = ["nu", "exp_l", "exp_j", "f", "exp_s"]

    for key, state in states_dict.items():
        # Create a new unique index for each state based on quantum numbers
        state["unique_id"] = state.apply(lambda row: "_".join([str(row[col]) for col in multi_index_columns]), axis=1)
        id_to_newid = dict(zip(state["id"], state["unique_id"], strict=True))
        id_to_n = dict(zip(state["id"], state["n"], strict=True))

        # Map the ids in the matrix elements table to the new unique identifiers
        table = table_dict[key]
        for which in ["initial", "final"]:
            table[f"newid_{which}"] = table[f"id_{which}"].map(id_to_newid)

            # Set new colum n and filter by min_n
            table[f"n_{which}"] = table[f"id_{which}"].map(id_to_n)
            if min_n > 1:
                table = table[table[f"n_{which}"] >= min_n].copy()

        # Set new colum with delta n = abs(n_final - n_initial) and filter by max_delta_n
        table["delta_n"] = (table["n_final"] - table["n_initial"]).abs()
        table = table[table["delta_n"] <= max_delta_n].copy()

        # Index the matrix elements by the newid
        table = table.set_index(["newid_initial", "newid_final"])
        table_dict[key] = table.sort_index()

    # Compare val values within tolerance
    new, old = table_dict["new"], table_dict["old"]
    val_diff = (new["val"] - old["val"]).abs()
    tolerance = atol + rtol * old["val"].abs()
    val_mask = val_diff.gt(tolerance)

    print(f"  Found {val_mask.sum()} val differences outside tolerance:")
    if verbose and val_mask.any():
        diff_uids = val_mask.loc[val_mask].index
        for uid in diff_uids:
            new_val = new.loc[uid, "val"]
            old_val = old.loc[uid, "val"]
            diff_val = val_diff[uid]
            print(f"    State initial: {uid[0]}; State final: {uid[1]}")
            print(f"      New val: {new_val}")
            print(f"      Old val: {old_val:.12f}")
            print(f"      Absolute difference: {diff_val:.2e}")
            print(f"      Relative difference: {diff_val / abs(old_val):.2e}")

    max_rdiff = (val_diff / old["val"].abs()).max()
    print(f"  Maximum absolute val difference: {val_diff.max():.2e}")
    print(f"  Maximum relative val difference: {max_rdiff:.2e}")


if __name__ == "__main__":
    main()

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
    name = "mqdt/Yb174_mqdt"
    old_path = Path(f"{name}_v1.1")
    new_path = Path(f"{name}_v1.2")

    print(f"Comparing matrix elements tables:\n  New: {new_path}\n  Old: {old_path}")
    for table_name in TABLE_NAMES:
        if not (new_path / f"{table_name}.parquet").exists() or not (old_path / f"{table_name}.parquet").exists():
            print(f"\nSkipping {table_name} as it does not exist in either the new or the old path.")
            continue
        compare_matrix_elements_table(
            table_name,
            new_path,
            old_path,
            max_delta_n=3,
            min_n=16,
            max_n=80,
            verbose=False,
            only_compare_absolute_values=False,
        )


def compare_matrix_elements_table(  # noqa: C901
    table_name: str,
    new_path: Path,
    old_path: Path,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    *,
    max_delta_n: int = 3,
    min_n: int = 1,
    max_n: int = 999,
    only_compare_absolute_values: bool = False,
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

    for key, table in table_dict.items():
        print(f"  {key.capitalize()} table shape: {table.shape} with columns: {list(table.columns)}")

    multi_index_columns = ["n", "exp_l", "exp_j"]
    if "mqdt" in str(new_path):
        multi_index_columns = ["nu", "exp_l", "exp_j", "f", "exp_s"]
        # round index columns to avoid floating point issues
        for col in multi_index_columns:
            decimals = {"nu": 1}.get(col, 3)
            states_dict["new"][col] = states_dict["new"][col].round(decimals)
            states_dict["old"][col] = states_dict["old"][col].round(decimals)

    for key, state in states_dict.items():
        # Create a new unique index for each state based on quantum numbers
        state["unique_id"] = state.apply(lambda row: "_".join([str(row[col]) for col in multi_index_columns]), axis=1)
        id_to_newid = dict(zip(state["id"], state["unique_id"], strict=True))
        id_to_n = dict(zip(state["id"], state["n"], strict=True))

        # Map the ids in the matrix elements table to the new unique identifiers
        table = table_dict[key]
        for which in ["initial", "final"]:
            table[f"newid_{which}"] = table[f"id_{which}"].map(id_to_newid)

        # Set new column n_... and filter by it
        for which in ["initial", "final"]:
            table[f"n_{which}"] = table[f"id_{which}"].map(id_to_n)
            table_dict[key] = table = table[table[f"n_{which}"] >= min_n].copy()
            table_dict[key] = table = table[table[f"n_{which}"] <= max_n].copy()

        # Set new column with delta n = abs(n_final - n_initial) and filter by it
        table["delta_n"] = (table["n_final"] - table["n_initial"]).abs()
        table_dict[key] = table = table[table["delta_n"] <= max_delta_n].copy()

        # Index the matrix elements by the newid
        table.set_index(["newid_initial", "newid_final"], inplace=True, drop=False)  # noqa: PD002
        table.sort_index(inplace=True)  # noqa: PD002

    new, old = table_dict["new"], table_dict["old"]
    print(f"  Values in new table: {new.shape[0]}; Values in old table: {old.shape[0]}")
    common_index = new.index.intersection(old.index)

    for table in table_dict.values():
        # only keep rows that are in both tables
        table.drop(index=common_index.symmetric_difference(table.index), inplace=True, errors="ignore")  # noqa: PD002
        table.sort_index(inplace=True)  # noqa: PD002
    print(f"  Common values: {new.shape[0]}")

    # Compare val values within tolerance
    if only_compare_absolute_values:
        val_diff = (new["val"].abs() - old["val"].abs()).abs()
    else:
        val_diff = (new["val"] - old["val"]).abs()
    tolerance = atol + rtol * old["val"].abs()
    val_mask = val_diff.gt(tolerance)

    print(f"  Found {val_mask.sum()}/{val_mask.shape[0]} val differences outside tolerance:")
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

    rdiff = val_diff / old["val"].abs()
    print(f"  Maximum absolute val difference: {val_diff.max():.2e}")
    print(f"  Min/Max relative val difference: {rdiff.min():.2e} / {rdiff.max():.2e}")


if __name__ == "__main__":
    main()

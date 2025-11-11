from pathlib import Path
from typing import Literal

import pandas as pd

COLUMNS: list[str] = [
    "parity",
    "n",
    "f",
    "exp_l",
    "exp_j",
    "exp_s",
    "exp_l_ryd",
    "exp_j_ryd",
    "std_nui",
    "std_l",
    "std_j",
    "std_s",
    "std_l_ryd",
    "std_j_ryd",
    "is_j_total_momentum",
    "is_calculated_with_mqdt",
    "underspecified_channel_contribution",
    "energy",
    "nu",
    "exp_nui",
]


def main() -> None:
    # CHANGE THESE PATHS, TO THE FOLDERS YOU WANT TO COMPARE
    name = "sqdt/"
    old_path = Path(f"{name}Sr88_triplet_v1.2")
    new_path = Path(f"{name}Sr88_sqdt_v1.3")

    compare_states_table(new_path, old_path, min_n=1, compare_id=False, verbosity="none")


def compare_states_table(  # noqa: C901, PLR0912, PLR0915
    new_path: Path,
    old_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-10,
    *,
    min_n: int = 1,
    max_n: int = 90,
    compare_id: bool = False,
    verbosity: Literal["none", "all", "diff", "states"],
) -> None:
    """Compare the states table of two versions of the database.

    Given two parquet files containing the states table, with the following columns:
    - n: principal quantum number
    - exp_l: experimental orbital angular momentum
    - exp_j: experimental total angular momentum
    - energy: energy of the state
    - etc.
    This function will compare the tables by
    i) checking if both tables include the same states (a state is defined by the combination of n, exp_l, and exp_j)
    ii) checking if the energy values of the states are equal within a given tolerance (rtol and atol).
    iii) checking if all other columns are exactly equal.

    """
    print(f"Comparing states tables:\n  New: {new_path}\n  Old: {old_path}\n")

    states_dict = {
        "new": pd.read_parquet(new_path / "states.parquet"),
        "old": pd.read_parquet(old_path / "states.parquet"),
    }
    for key, states in states_dict.items():
        states_dict[key] = states[states["n"] >= min_n]
        states_dict[key] = states[states["n"] <= max_n]

    for key, states in states_dict.items():
        print(f"{key.capitalize()} states table:")
        print(f"  Table shape: {states.shape}; Columns: {list(states.columns)}\n")

    missing_cols = [col for col in COLUMNS if col not in states_dict["new"].columns]
    if len(missing_cols) > 0:
        raise ValueError(f"New states table is missing columns: {missing_cols}")
    missing_cols = [col for col in states_dict["new"].columns if col not in COLUMNS and col != "id"]
    if len(missing_cols) > 0:
        print(f"Warning: New states table has extra columns: {missing_cols}")

    multi_index_columns = ["n", "exp_l", "exp_j", "exp_s"]
    if "mqdt" in str(new_path):
        multi_index_columns = ["nu", "exp_l", "exp_j", "f", "exp_s"]
        # round index columns to avoid floating point issues
        for states in states_dict.values():
            for col in multi_index_columns:
                decimals = {"nu": 1, "energy": 6}.get(col, 3)
                states[col] = states[col].round(decimals)

    for key, states in states_dict.items():
        states_dict[key] = states.set_index(multi_index_columns, verify_integrity=True, drop=False)

    # Check if both tables have the same states
    new, old = states_dict["new"], states_dict["old"]
    new_states = set(new.index)
    old_states = set(old.index)
    only_in_new = new_states - old_states
    only_in_old = old_states - new_states

    if only_in_new or only_in_old:
        print("States that don't match between tables:")
        if only_in_new:
            print(f"  {len(only_in_new)} states only in new table:")
            if verbosity in ["all", "states"]:
                for state in sorted(only_in_new):
                    print(", ".join(f"{col}={state[i]}" for i, col in enumerate(multi_index_columns)))
        if only_in_old:
            print(f"  {len(only_in_old)} states only in old table:")
            if verbosity in ["all", "states"]:
                for state in sorted(only_in_old):
                    print(", ".join(f"{col}={state[i]}" for i, col in enumerate(multi_index_columns)))

        # Remove non-matching states from both tables
        new = new.drop(index=list(only_in_new), errors="ignore")
        old = old.drop(index=list(only_in_old), errors="ignore")

    if len(new) == 0:
        print("No matching states to compare...")
    if len(new) != len(old):
        print(f"Warning: Number of matching states differs: {len(new)} vs {len(old)}")
    print(f"Continuing comparison with {len(new)} matching states ...\n")

    # Compare all columns except energy
    compare_with_tolerance = ["energy", "nu", "exp_nui", "exp_j_ryd", "std_j_ryd"]

    for col in COLUMNS:
        if col in compare_with_tolerance:
            continue
        if not compare_id and col == "id":
            continue

        differences = new[col].ne(old[col])
        if not differences.any():
            print(f"No differences found in column '{col}'.")
            continue
        print(f"Found {differences.sum()} differences in column '{col}':")
        if verbosity in ["all", "diff"]:
            diff_states = differences.loc[differences].index
            for state in diff_states:
                state_str = ", ".join([f"{col}={state[i]}" for i, col in enumerate(multi_index_columns)])
                print(f"  State {state_str}:")
                print(f"    New value: {new.loc[state, col]}")
                print(f"    Old value: {old.loc[state, col]}")
    print()

    # Compare energy values within tolerance
    for col in COLUMNS:
        if col not in compare_with_tolerance:
            continue
        differences = (new[col] - old[col]).abs()
        tolerance = atol + rtol * old[col].abs()
        mask = differences.gt(tolerance)

        print(f"Found {mask.sum()} {col} differences outside tolerance:")
        if verbosity in ["all", "diff"] and mask.any():
            diff_states = mask.loc[mask].index
            for state in diff_states:
                new = new.loc[state, col]
                old = old.loc[state, col]
                diff = differences[state]
                state_str = ", ".join([f"{col}={state[i]}" for i, col in enumerate(multi_index_columns)])
                print(f"  State {state_str}:")
                print(f"    New {col}: {new:.12f}")
                print(f"    Old {col}: {old:.12f}")
                print(f"    Absolute difference: {diff:.2e}")
                print(f"    Relative difference: {diff / abs(old):.2e}")

        max_rdiff = (differences / old[col].abs()).max()
        print(f"Maximum absolute {col} difference: {differences.max():.2e}")
        print(f"Maximum relative {col} difference: {max_rdiff:.2e}")


if __name__ == "__main__":
    main()

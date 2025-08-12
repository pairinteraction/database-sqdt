from pathlib import Path

import pandas as pd


def main() -> None:
    # CHANGE THESE PATHS, TO THE FOLDERS YOU WANT TO COMPARE
    name = "Sr88_singlet"
    new_path = Path(f"{name}_v1.2")
    old_path = Path(f"{name}_v1.1")

    compare_states_table(new_path, old_path, min_n=1, compare_id=False, verbose=False)


def compare_states_table(  # noqa: C901, PLR0912, PLR0915
    new_path: Path,
    old_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-10,
    *,
    min_n: int = 1,
    compare_id: bool = False,
    verbose: bool = False,
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

    new = pd.read_parquet(new_path / "states.parquet")
    if min_n > 1:
        new = new[new["n"] >= min_n].copy()
    new = new.set_index(["n", "exp_l", "exp_j"], verify_integrity=True)

    old = pd.read_parquet(old_path / "states.parquet")
    if min_n > 1:
        old = old[old["n"] >= min_n].copy()
    old = old.set_index(["n", "exp_l", "exp_j"], verify_integrity=True)

    # Check if both tables have the same states
    new_states = set(new.index)
    old_states = set(old.index)
    only_in_new = new_states - old_states
    only_in_old = old_states - new_states

    if only_in_new or only_in_old:
        print("States that don't match between tables:")
        if only_in_new:
            print(f"  {len(only_in_new)} states only in new table:")
            if verbose:
                for state in sorted(only_in_new):
                    print(f"    n={state[0]}, l={state[1]}, j={state[2]}")
        if only_in_old:
            print(f"  {len(only_in_old)} states only in old table:")
            if verbose:
                for state in sorted(only_in_old):
                    print(f"    n={state[0]}, l={state[1]}, j={state[2]}")

        # Remove non-matching states from both tables
        new = new.drop(index=list(only_in_new), errors="ignore")
        old = old.drop(index=list(only_in_old), errors="ignore")
        print("Continuing comparison with matching states only...\n")

    # Compare all columns except energy
    columns_to_compare = [col for col in new.columns if col != "energy"]

    for col in columns_to_compare:
        if not compare_id and col == "id":
            continue

        differences = new[col].ne(old[col])
        if differences.any():
            print(f"Found {differences.sum()} differences in column '{col}':")
            if verbose:
                diff_states = differences.loc[differences].index
                for state in diff_states:
                    print(f"  State (n={state[0]}, l={state[1]}, j={state[2]}):")
                    print(f"    New value: {new.loc[state, col]}")
                    print(f"    Old value: {old.loc[state, col]}")
    print()

    # Compare energy values within tolerance
    energy_diff = (new["energy"] - old["energy"]).abs()
    tolerance = atol + rtol * old["energy"].abs()
    energy_mask = energy_diff.gt(tolerance)

    print(f"Found {energy_mask.sum()} energy differences outside tolerance:")
    if verbose and energy_mask.any():
        diff_states = energy_mask.loc[energy_mask].index
        for state in diff_states:
            new_energy = new.loc[state, "energy"]
            old_energy = old.loc[state, "energy"]
            diff_energy = energy_diff[state]
            print(f"  State (n={state[0]}, l={state[1]}, j={state[2]}):")
            print(f"    New energy: {new_energy:.12f}")
            print(f"    Old energy: {old_energy:.12f}")
            print(f"    Absolute difference: {diff_energy:.2e}")
            print(f"    Relative difference: {diff_energy / abs(old_energy):.2e}")

    max_rdiff = (energy_diff / old["energy"].abs()).max()
    print(f"Maximum absolute energy difference: {energy_diff.max():.2e}")
    print(f"Maximum relative energy difference: {max_rdiff:.2e}")


if __name__ == "__main__":
    main()

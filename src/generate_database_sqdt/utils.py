from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import ryd_numerov
from ryd_numerov import RydbergState
from ryd_numerov.angular import calc_reduced_angular_matrix_element
from ryd_numerov.elements import Element
from ryd_numerov.radial import calc_radial_matrix_element

if TYPE_CHECKING:
    from ryd_numerov.units import OperatorType


def get_sorted_list_of_states(species: str, n_min: int, n_max: int) -> list[RydbergState]:
    """Create a list of quantum numbers sorted by their state energies."""
    element = ryd_numerov.elements.Element.from_species(species)
    list_of_states: list[RydbergState] = []
    for n in range(n_min, n_max + 1):
        for l in range(n):
            if not element.is_allowed_shell(n, l):
                continue
            for j in np.arange(abs(l - element.s), l + element.s + 1):
                list_of_states.append(RydbergState(species, n, l, float(j)))  # noqa: PERF401
    return sorted(list_of_states, key=lambda x: x.get_energy("a.u."))


def calc_matrix_element_one_pair(
    species: str,
    n1: int,
    l1: int,
    j1: float,
    n2: int,
    l2: int,
    j2: float,
    matrix_elements_of_interest: dict[str, tuple["OperatorType", int, int]],
) -> dict[str, float]:
    element = Element.from_species(species)

    matrix_elements: dict[str, float] = {}
    for tkey, (operator, k_radial, k_angular) in matrix_elements_of_interest.items():
        angular_matrix_element_au = calc_reduced_angular_matrix_element_cached(
            element.s, l1, j1, element.s, l2, j2, operator, k_angular
        )
        if angular_matrix_element_au == 0:
            continue

        radial_matrix_element_au = calc_radial_matrix_element_cached(species, n1, l1, j1, n2, l2, j2, k_radial)
        if radial_matrix_element_au == 0:
            continue

        # - mu_B in atomic units = -1/2; (e in atomic units = 1)
        prefactor = -1 / 2 if operator == "MAGNETIC" else 1
        matrix_elements[tkey] = prefactor * radial_matrix_element_au * angular_matrix_element_au

    return matrix_elements


# since we sort the states by l, before calculating the matrix elements a rather low number of cache size is sufficient
@lru_cache(maxsize=2_000)
def calc_reduced_angular_matrix_element_cached(
    s1: int, l1: int, j1: float, s2: int, l2: int, j2: float, operator: "OperatorType", k_angular: int
) -> float:
    return calc_reduced_angular_matrix_element(s1, l1, j1, s2, l2, j2, operator, k_angular)


# this cache is basically only used within one call of get_matrix_element
@lru_cache(maxsize=100)
def calc_radial_matrix_element_cached(
    species: str, n1: int, l1: int, j1: float, n2: int, l2: int, j2: float, k_radial: int
) -> float:
    if (n1, l1, j1) > (n2, l2, j2):  # for better use of the cache
        return calc_radial_matrix_element_cached(species, n2, l2, j2, n1, l1, j1, k_radial)

    state1 = get_rydberg_state_cached(species, n1, l1, j1)
    state2 = get_rydberg_state_cached(species, n2, l2, j2)
    return calc_radial_matrix_element(state1, state2, k_radial)


# Cache size should be one the order of N_MAX * 4 * 2
# (since for each initial state we loop over all l' = l, l+1, l+2 and l+3 final states (and all j final))
@lru_cache(maxsize=2_000)
def get_rydberg_state_cached(species: str, n: int, l: int, j: float) -> RydbergState:
    """Get the cached rydberg state (where the wavefunction was already calculated)."""
    state = RydbergState(species, n, l, j)
    state.create_wavefunction()
    return state

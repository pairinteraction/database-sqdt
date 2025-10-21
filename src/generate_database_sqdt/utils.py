from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from ryd_numerov import RydbergStateAlkali
from ryd_numerov.angular import AngularKetLS
from ryd_numerov.elements import BaseElement

if TYPE_CHECKING:
    from ryd_numerov.angular.angular_matrix_element import AngularOperatorType
    from ryd_numerov.units import MatrixElementType

OPERATOR_TO_KS = {  # operator: (k_radial, k_angular)
    "MAGNETIC_DIPOLE": (0, 1),
    "ELECTRIC_DIPOLE": (1, 1),
    "ELECTRIC_QUADRUPOLE": (2, 2),
    "ELECTRIC_OCTUPOLE": (3, 3),
    "ELECTRIC_QUADRUPOLE_ZERO": (2, 0),
}


@lru_cache(maxsize=10)
def element_from_species(species: str) -> BaseElement:
    """Get the BaseElement from the species string."""
    return BaseElement.from_species(species)


def get_sorted_list_of_states(species: str, n_min: int, n_max: int) -> list[RydbergStateAlkali]:
    """Create a list of quantum numbers sorted by their state energies."""
    element = element_from_species(species)
    list_of_states: list[RydbergStateAlkali] = []
    for n in range(n_min, n_max + 1):
        for l in range(n):
            if not element.is_allowed_shell(n, l, 1 / 2):
                continue
            for j in np.arange(abs(l - 1 / 2), l + 1 / 2 + 1):
                state = RydbergStateAlkali(species, n, l, float(j))
                state.create_element(use_nist_data=True)
                list_of_states.append(state)
    return sorted(list_of_states, key=lambda x: x.get_energy("a.u."))


def calc_matrix_element_one_pair(
    species: str,
    n1: int,
    l1: int,
    j1: float,
    n2: int,
    l2: int,
    j2: float,
    matrix_elements_of_interest: dict[str, "MatrixElementType"],
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, operator in matrix_elements_of_interest.items():
        k_radial, k_angular = OPERATOR_TO_KS[operator]

        if operator == "MAGNETIC_DIPOLE":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = calc_reduced_angular_matrix_element_cached(l1, j1, l2, j2, "s_tot", k_angular, species)
            g_l = 1
            value_l_tot = calc_reduced_angular_matrix_element_cached(l1, j1, l2, j2, "l_tot", k_angular, species)
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot
            prefactor = -0.5  # - mu_B in atomic units

        elif operator in ["ELECTRIC_DIPOLE", "ELECTRIC_QUADRUPOLE", "ELECTRIC_OCTUPOLE", "ELECTRIC_QUADRUPOLE_ZERO"]:
            angular_matrix_element = calc_reduced_angular_matrix_element_cached(
                l1, j1, l2, j2, "SPHERICAL", k_angular, species
            )
            prefactor = np.sqrt(4 * np.pi / (2 * k_angular + 1))  # e in atomic units is 1
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if angular_matrix_element == 0:
            continue

        radial_matrix_element_au = calc_radial_matrix_element_cached(species, n1, l1, j1, n2, l2, j2, k_radial)
        if radial_matrix_element_au == 0:
            continue

        matrix_elements[tkey] = prefactor * radial_matrix_element_au * angular_matrix_element

    return matrix_elements


# since we sort the states by l, before calculating the matrix elements a rather low number of cache size is sufficient
@lru_cache(maxsize=2_000)
def calc_reduced_angular_matrix_element_cached(
    l1: int,
    j1: float,
    l2: int,
    j2: float,
    operator: "AngularOperatorType",
    k_angular: int,
    species: str,
) -> float:
    ket1 = AngularKetLS(l_r=l1, j_tot=j1, species=species)
    ket2 = AngularKetLS(l_r=l2, j_tot=j2, species=species)
    return ket2.calc_reduced_matrix_element(ket1, operator, k_angular)


def calc_radial_matrix_element_cached(
    species: str, n1: int, l1: int, j1: float, n2: int, l2: int, j2: float, k_radial: int
) -> float:
    # if l is so large, that there is no quantum defect anymore,
    # then the radial wavefunction is the same for all j, so we set j = l - 1/2
    max_l = get_max_l_with_quantum_defect(species)
    j1 = l1 - 1 / 2 if l1 > max_l else j1
    j2 = l2 - 1 / 2 if l2 > max_l else j2

    if k_radial == 0 and (l1, j1) == (l2, j2):
        return 1 if n1 == n2 else 0

    if (n1, l1, j1) > (n2, l2, j2):  # for better use of the cache
        return _calc_radial_matrix_element_cached(species, n2, l2, j2, n1, l1, j1, k_radial)

    return _calc_radial_matrix_element_cached(species, n1, l1, j1, n2, l2, j2, k_radial)


# Cache size should be at least on the order of 4 * (all_n_up_to + 2 * max_delta_n)
# however, for the first n until n=all_n_up_to we need an even larger cache size
@lru_cache(maxsize=50_000)
def _calc_radial_matrix_element_cached(
    species: str, n1: int, l1: int, j1: float, n2: int, l2: int, j2: float, k_radial: int
) -> float:
    state1 = get_rydberg_state_cached(species, n1, l1, j1)
    state2 = get_rydberg_state_cached(species, n2, l2, j2)
    return state1.radial.calc_matrix_element(state2.radial, k_radial, unit="a.u.")


@lru_cache(maxsize=10)
def get_max_l_with_quantum_defect(species: str) -> int:
    """Get the maximum l with quantum defect for a given species."""
    element = element_from_species(species)
    return max([l for (l, *_) in element._quantum_defects], default=0)  # noqa: SLF001


# Cache size should be one the order of N_MAX * 4 * 2
# (since for each initial state we loop over all l' = l, l+1, l+2 and l+3 final states (and all j final))
@lru_cache(maxsize=2_000)
def get_rydberg_state_cached(species: str, n: int, l: int, j: float) -> RydbergStateAlkali:
    """Get the cached rydberg state (where the wavefunction was already calculated)."""
    state = RydbergStateAlkali(species, n, l, j)
    state.create_element(use_nist_data=True)
    state.radial.create_wavefunction(sign_convention="n_l_1")
    return state

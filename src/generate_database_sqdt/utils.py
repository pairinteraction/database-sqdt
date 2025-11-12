from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from rydstate import RydbergStateAlkali, RydbergStateAlkalineJJ, RydbergStateAlkalineLS
from rydstate.angular import AngularKetJJ, AngularKetLS
from rydstate.radial import RadialState
from rydstate.species import SpeciesObject
from rydstate.units import MatrixElementOperatorRanks

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import CouplingScheme
    from rydstate.angular.angular_matrix_element import AngularOperatorType
    from rydstate.units import MatrixElementOperator


MIN_L_TO_USE_JJ_COUPLING = 6


def get_sorted_list_of_states(
    species_name: str, n_min: int, n_max: int
) -> list[RydbergStateAlkali | RydbergStateAlkalineLS | RydbergStateAlkalineJJ]:
    """Create a list of quantum numbers sorted by their state energies."""
    species = SpeciesObject.from_name(species_name)
    if species.number_valence_electrons == 1:
        return _get_sorted_list_of_states_alkali(species, n_min, n_max)
    if species.number_valence_electrons == 2:  # noqa: PLR2004
        return _get_sorted_list_of_states_alkaline(species, n_min, n_max)

    raise NotImplementedError("Only species with 1 or 2 valence electrons are supported.")


def _get_sorted_list_of_states_alkali(species: SpeciesObject, n_min: int, n_max: int) -> list[RydbergStateAlkali]:
    s = 1 / 2
    i_c = species.i_c if species.i_c is not None else 0
    list_of_states: list[RydbergStateAlkali] = []
    for n in range(n_min, n_max + 1):
        for l in range(n):
            if not species.is_allowed_shell(n, l, s):
                continue
            for j in np.arange(abs(l - s), l + s + 1):
                for f in np.arange(abs(j - i_c), j + i_c + 1):
                    state = RydbergStateAlkali(species, n, l, float(j), f=float(f))
                    list_of_states.append(state)

    return sorted(list_of_states, key=lambda x: x.get_energy("a.u."))


def _get_sorted_list_of_states_alkaline(  # noqa: C901, PLR0912
    species: SpeciesObject, n_min: int, n_max: int
) -> list[RydbergStateAlkalineLS | RydbergStateAlkalineJJ]:
    i_c = species.i_c if species.i_c is not None else 0
    s_r, s_c = 1 / 2, 1 / 2
    list_of_states: list[RydbergStateAlkalineLS | RydbergStateAlkalineJJ] = []
    for n in range(n_min, n_max + 1):
        for l in range(n):
            if l < MIN_L_TO_USE_JJ_COUPLING:  # low-l states use LS coupling
                for s_tot in [0, 1]:
                    if not species.is_allowed_shell(n, l, s_tot):
                        continue
                    for j_tot in range(abs(l - s_tot), l + s_tot + 1):
                        for f in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                            state = RydbergStateAlkalineLS(species, n, l, s_tot=s_tot, j_tot=j_tot, f_tot=float(f))
                            list_of_states.append(state)

            else:  # high-l states use jj coupling
                if not all(species.is_allowed_shell(n, l, s_tot) for s_tot in [0, 1]):
                    if any(species.is_allowed_shell(n, l, s_tot) for s_tot in [0, 1]):
                        raise RuntimeError("JJ coupling singlet and triplet differ in is_allowed_shell check.")
                    continue
                for j_r in [l - s_r, l + s_r]:
                    if not (j_r + s_c).is_integer():
                        raise RuntimeError("Non-integer j_r + s_c encountered.")
                    for j_tot in range(int(abs(j_r - s_c)), int(j_r + s_c + 1)):
                        for f in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                            state = RydbergStateAlkalineJJ(species, n, l, j_r=j_r, j_tot=j_tot, f_tot=float(f))
                            list_of_states.append(state)

    return sorted(list_of_states, key=lambda x: x.get_energy("a.u."))


def calc_matrix_element_one_pair(
    state1: RydbergStateAlkali | RydbergStateAlkalineLS,
    state2: RydbergStateAlkali | RydbergStateAlkalineLS,
    matrix_elements_of_interest: dict[str, MatrixElementOperator],
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, operator in matrix_elements_of_interest.items():
        k_radial, k_angular = MatrixElementOperatorRanks[operator]

        if operator == "magnetic_dipole":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = calc_reduced_angular_matrix_element_cached(
                state1.angular.coupling_scheme,
                state1.angular.quantum_numbers,
                state2.angular.coupling_scheme,
                state2.angular.quantum_numbers,
                "s_tot",
                k_angular,
            )
            g_l = 1
            value_l_tot = calc_reduced_angular_matrix_element_cached(
                state1.angular.coupling_scheme,
                state1.angular.quantum_numbers,
                state2.angular.coupling_scheme,
                state2.angular.quantum_numbers,
                "l_tot",
                k_angular,
            )
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot
            prefactor = -0.5  # - mu_B in atomic units

        elif operator in ["electric_dipole", "electric_quadrupole", "electric_octupole", "electric_quadrupole_zero"]:
            angular_matrix_element = calc_reduced_angular_matrix_element_cached(
                state1.angular.coupling_scheme,
                state1.angular.quantum_numbers,
                state2.angular.coupling_scheme,
                state2.angular.quantum_numbers,
                "spherical",
                k_angular,
            )
            prefactor = math.sqrt(4 * math.pi / (2 * k_angular + 1))  # e in atomic units is 1
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if angular_matrix_element == 0:
            continue

        radial_matrix_element_au = calc_radial_matrix_element_cached(
            state1.species.name,
            *(state1.n, state1.get_nu(), state1.angular.l_r),
            *(state2.n, state2.get_nu(), state2.angular.l_r),
            k_radial,
        )
        if radial_matrix_element_au == 0:
            continue

        matrix_elements[tkey] = prefactor * radial_matrix_element_au * angular_matrix_element

    return matrix_elements


@lru_cache(maxsize=100_000)
def calc_reduced_angular_matrix_element_cached(
    coupling_scheme1: CouplingScheme,
    qns1: tuple[float, ...],
    coupling_scheme2: CouplingScheme,
    qns2: tuple[float, ...],
    operator: AngularOperatorType,
    k_angular: int,
) -> float:
    ket_classes = {"LS": AngularKetLS, "JJ": AngularKetJJ}
    ket1 = ket_classes[coupling_scheme1](*qns1)  # type: ignore[arg-type]
    ket2 = ket_classes[coupling_scheme2](*qns2)  # type: ignore[arg-type]
    return ket2.calc_reduced_matrix_element(ket1, operator, k_angular)


def calc_radial_matrix_element_cached(
    species_name: str, n1: int, nu1: float, l1: int, n2: int, nu2: float, l2: int, k_radial: int
) -> float:
    if k_radial == 0 and nu1 == nu2:
        return 1 if l1 == l2 else 0

    if (nu1, l1) > (nu2, l2):  # for better use of the cache
        return _calc_radial_matrix_element_cached(species_name, n2, nu2, l2, n1, nu1, l1, k_radial)

    return _calc_radial_matrix_element_cached(species_name, n1, nu1, l1, n2, nu2, l2, k_radial)


# Cache size should be at least on the order of 4 * (all_n_up_to + 2 * max_delta_n)
# however, for the first n until n=all_n_up_to we need an even larger cache size
@lru_cache(maxsize=50_000)
def _calc_radial_matrix_element_cached(
    species_name: str, n1: int, nu1: float, l1: int, n2: int, nu2: float, l2: int, k_radial: int
) -> float:
    state1 = get_radial_state_cached(species_name, n1, nu1, l1)
    state2 = get_radial_state_cached(species_name, n2, nu2, l2)
    return state1.calc_matrix_element(state2, k_radial, unit="a.u.")


# Cache size should be one the order of N_MAX * 4 * 2
# (since for each initial state we loop over all l' = l, l+1, l+2 and l+3 final states (and all j final))
@lru_cache(maxsize=2_000)
def get_radial_state_cached(species_name: str, n: int, nu: float, l: int) -> RadialState:
    """Get the cached rydberg state (where the wavefunction was already calculated)."""
    state = RadialState(species_name, nu, l)
    state.set_n_for_sanity_check(n)
    state.create_wavefunction(sign_convention="n_l_1")
    return state

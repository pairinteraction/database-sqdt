from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from ryd_numerov import RydbergState
from ryd_numerov.angular import calc_reduced_angular_matrix_element
from ryd_numerov.angular.utils import calc_wigner_3j, minus_one_pow
from ryd_numerov.elements import BaseElement
from ryd_numerov.radial import calc_radial_matrix_element

if TYPE_CHECKING:
    import numpy.typing as npt
    from ryd_numerov.units import OperatorType


@lru_cache(maxsize=10)
def element_from_species(species: str) -> BaseElement:
    """Get the BaseElement from the species string."""
    return BaseElement.from_species(species)


def get_sorted_list_of_states(species: str, n_min: int, n_max: int) -> list[RydbergState]:
    """Create a list of quantum numbers sorted by their state energies."""
    element = element_from_species(species)
    s_list: list[float] = {1: [1 / 2], 2: [0, 1]}.get(element.number_valence_electrons, [])  # type: ignore [assignment]
    list_of_states: list[RydbergState] = []
    for s_tot in s_list:
        for n in range(n_min, n_max + 1):
            for l in range(n):
                if not element.is_allowed_shell(n, l, s_tot):
                    continue
                for j_tot in np.arange(abs(l - s_tot), l + s_tot + 1):
                    state = RydbergState(species, n=n, l=l, j_tot=float(j_tot), s_tot=s_tot)
                    list_of_states.append(state)
    return sorted(list_of_states, key=lambda x: x.get_energy("a.u."))


def calc_matrix_element_one_pair(
    species: str,
    n1: int,
    l1: int,
    j1: float,
    s1: float,
    n2: int,
    l2: int,
    j2: float,
    s2: float,
    matrix_elements_of_interest: dict[str, tuple["OperatorType", int, int]],
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, (operator, k_radial, k_angular) in matrix_elements_of_interest.items():
        angular_matrix_element_au = calc_reduced_angular_matrix_element_cached(
            s1, l1, j1, s2, l2, j2, operator, k_angular
        )

        if angular_matrix_element_au == 0:
            continue

        radial_matrix_element_au = calc_radial_matrix_element_cached(species, n1, l1, j1, s1, n2, l2, j2, s2, k_radial)
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


def calc_radial_matrix_element_cached(
    species: str, n1: int, l1: int, j1: float, s1: float, n2: int, l2: int, j2: float, s2: float, k_radial: int
) -> float:
    # if l is so large, that there is no quantum defect anymore,
    # then the radial wavefunction is the same for all j_tot and s_tot, so we set j_tot = l - (s_tot % 1)
    element = element_from_species(species)
    max_l = get_max_l_with_quantum_defect(species)
    j_shift = (element.number_valence_electrons / 2) % 1
    j1 = l1 - j_shift if l1 > max_l else j1
    j2 = l2 - j_shift if l2 > max_l else j2

    if k_radial == 0 and (l1, j1) == (l2, j2):
        return 1 if n1 == n2 else 0

    qns1, qns2 = (n1, l1, j1, s1), (n2, l2, j2, s2)
    if qns1 > qns2:  # for better use of the cache
        return _calc_radial_matrix_element_cached(species, *qns2, *qns1, k_radial)

    return _calc_radial_matrix_element_cached(species, *qns1, *qns2, k_radial)


# Cache size should be at least on the order of 4 * (all_n_up_to + 2 * max_delta_n)
# however, for the first n until n=all_n_up_to we need an even larger cache size
@lru_cache(maxsize=50_000)
def _calc_radial_matrix_element_cached(
    species: str, n1: int, l1: int, j1: float, s1: float, n2: int, l2: int, j2: float, s2: float, k_radial: int
) -> float:
    state1 = get_rydberg_state_cached(species, n1, l1, j1, s1)
    state2 = get_rydberg_state_cached(species, n2, l2, j2, s2)
    return calc_radial_matrix_element(state1, state2, k_radial)


@lru_cache(maxsize=10)
def get_max_l_with_quantum_defect(species: str) -> int:
    """Get the maximum l with quantum defect for a given species."""
    element = element_from_species(species)
    return max([l for (l, *_) in element._quantum_defects], default=0)  # noqa: SLF001


# Cache size should be one the order of N_MAX * 4 * 2
# (since for each initial state we loop over all l' = l, l+1, l+2 and l+3 final states (and all j final))
@lru_cache(maxsize=2_000)
def get_rydberg_state_cached(species: str, n: int, l: int, j_tot: float, s_tot: float) -> RydbergState:
    """Get the cached rydberg state (where the wavefunction was already calculated)."""
    state = RydbergState(species, n=int(n), l=int(l), j_tot=float(j_tot), s_tot=float(s_tot))
    state.create_wavefunction(sign_convention="n_l_1")
    return state


@lru_cache(maxsize=10_000_000)
def calc_wigner_3j_cached(j_1: float, j_2: float, j_3: float, m_1: float, m_2: float, m_3: float) -> float:
    if not j_1 <= j_2 <= j_3:  # better use of caching
        args_nd = np.array([j_1, j_2, j_3, m_1, m_2, m_3])
        inds = np.argsort(args_nd[:3])
        wigner = calc_wigner_3j_cached(*args_nd[:3][inds], *args_nd[3:][inds])
        if (inds[1] - inds[0]) in [1, -2]:
            return wigner
        return minus_one_pow(j_1 + j_2 + j_3) * wigner

    if m_3 < 0 or (m_3 == 0 and m_2 < 0):  # better use of caching
        return minus_one_pow(j_1 + j_2 + j_3) * calc_wigner_3j_cached(j_1, j_2, j_3, -m_1, -m_2, -m_3)

    return calc_wigner_3j(j_1, j_2, j_3, m_1, m_2, m_3)


def filter_qns(
    qns_list: "npt.NDArray[np.floating]",
    qns_ref: tuple[int, int, int, float, float],
    all_n_up_to: int,
    max_delta_n: int,
    k_angular_max: int,
) -> "npt.NDArray[np.floating]":
    """Filter the list of quantum numbers based on selection rules."""
    mask = _filter_qns_njit(
        n_list=qns_list[:, 1],
        l_list=qns_list[:, 2],
        s_list=qns_list[:, 4],
        n_ref=qns_ref[1],
        l_ref=qns_ref[2],
        s_ref=qns_ref[4],
        all_n_up_to=all_n_up_to,
        max_delta_n=max_delta_n,
        k_angular_max=k_angular_max,
    )

    return qns_list[mask]  # type: ignore [no-any-return]


@njit(cache=True)
def _filter_qns_njit(
    n_list: "npt.NDArray[np.integer]",
    l_list: "npt.NDArray[np.integer]",
    s_list: "npt.NDArray[np.floating]",
    n_ref: int,
    l_ref: int,
    s_ref: float,
    all_n_up_to: int,
    max_delta_n: int,
    k_angular_max: int,
) -> "npt.NDArray[np.bool]":
    mask = [True] * len(n_list)
    for i, (n, l, s) in enumerate(zip(n_list, l_list, s_list)):  # noqa: B905
        if n > all_n_up_to and n_ref > all_n_up_to and abs(n - n_ref) > max_delta_n:
            # If delta_n is larger than max_delta_n, we dont calculate the matrix elements anymore,
            # since these are so small, that they are usually not relevant for further calculations
            # However, we keep all dipole interactions with small n (we choose all_n_up_to as a cutoff)
            # since these are relevant for the spontaneous decay rates
            mask[i] = False
            continue
        if abs(l - l_ref) > k_angular_max:
            # if delta_l is larger than k_angular_max there is no matrix element we calculate
            mask[i] = False
            continue
        if abs(s - s_ref) != 0:
            # if delta_s is not 0 the matrix element is anyway 0
            mask[i] = False
            continue
    return np.array(mask)

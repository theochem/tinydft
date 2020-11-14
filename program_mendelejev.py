#!/usr/bin/env python3
# Tiny DFT is a minimalistic atomic DFT implementation.
# Copyright (C) 2019 The Tiny DFT Development Team
#
# This file is part of Tiny DFT.
#
# Tiny DFT is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Tiny DFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Compute atomic DFT results for a series of elements."""

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from tinybasis import Basis
from tinydft import scf_atom, build_rho
from tinygrid import setup_grid


__all__ = ['main', 'char2angqn', 'interpret_econf', 'klechkowski']


def main():
    """Compute atomic DFT results for a series of elements."""
    for atnum in range(1, 19):
        plot_atom(atnum, 0)


def plot_atom(atnum: float, atcharge: float):
    """Run an atomic DFT calculation and make a nice plot of the density.

    Parameters
    ----------
    atnum
        Atomic number.
    atcharge
        The net charge.

    """
    nelec = atnum - atcharge
    econf = klechkowski(nelec)
    occups = interpret_econf(econf)
    print("Nuclear charge                    {:8.1f}".format(atnum))
    print("Electronic configuration          {:s}".format(str(econf)))

    grid = setup_grid()
    basis = Basis(grid)
    # Compute the overlap matrix, just to check the numerics of the basis.
    evals_olp = np.linalg.eigvalsh(basis.olp)
    print("Number of radial grid points      {:8d}".format(len(grid.points)))
    print("Number of basis functions         {:8d}".format(basis.nbasis))
    print("Condition number of the overlap   {:8.1e}".format(evals_olp[-1] / evals_olp[0]))

    energies, eps_orbs_u = scf_atom(atnum, occups, grid, basis)

    # Construct the spin-summed density
    rho = build_rho(occups, eps_orbs_u, grid, basis)

    # Construct alpha and beta density.
    # Keep in mind that these are only constructed post-hoc. Internally, the
    # atomic DFT results, specifically the exchange-correlation term, is
    # computed for a spin-unpalized density.
    occups_alpha = []
    occups_beta = []
    maxangqn = len(occups) - 1
    for angqn in range(maxangqn + 1):
        occ_max = angqn * 2 + 1
        occups_alpha.append(np.clip(occups[angqn], 0, occ_max))
        occups_beta.append(np.clip(occups[angqn] - occups_alpha[angqn], 0, occ_max))
    rho_alpha = build_rho(occups_alpha, eps_orbs_u, grid, basis)
    rho_beta = build_rho(occups_beta, eps_orbs_u, grid, basis)

    # Construct noble core density
    noble_core_id = np.searchsorted(NOBLE_ATNUMS, nelec, side="left") - 1
    if noble_core_id >= 0:
        atnum_core = NOBLE_ATNUMS[noble_core_id]
        occups_core = interpret_econf(klechkowski(atnum_core))
        rho_core = build_rho(occups_core, eps_orbs_u, grid, basis)
    else:
        rho_core = np.zeros_like(rho)

    # Get the HOMO orbital energy
    eps_homo = max(
        evals[occups[angqn] > 0].max()
        for angqn, (evals, _evecs) in enumerate(eps_orbs_u))

    # Make a nice plot of the densities.
    plt.rcParams['font.sans-serif'] = ["Fira Sans"]
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.text(1, 1, format_econf(econf), ha="right", va="top", transform=ax.transAxes)
    ax.text(
        1, 0.85,
        (f"Z={atnum}" "\n"
         fr"$E_{{\mathrm{{KS}}}}$ = {energies[0]:.2f} $\mathrm{{E_h}}$" "\n"
         fr"-$\epsilon_N$ = {-eps_homo:.2f} $\mathrm{{E_h}}$"),
        ha="right", va="top", transform=ax.transAxes)
    ax.text(1, 0.5, SYMBOLS[atnum], ha="right", va="top", transform=ax.transAxes, fontsize=28)
    ax.semilogy(grid.points, rho_core / 2, "k-", alpha=0.25)
    ax.semilogy(grid.points, rho_beta, "-", color="C0", lw=3, alpha=1.0)
    ax.semilogy(grid.points, rho_alpha, "-", color="C1", alpha=1.0)
    ax.semilogy(grid.points, rho, "k:", color="C7")
    ax.set_xlabel(r"Distance [$\mathrm{a_0}$]")
    ax.set_ylabel(r"Density [$1/\mathrm{a_0}^3$]")
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlim(0, 6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("rho_z{:03d}_{}.png".format(atnum, '_'.join(econf.split())), dpi=300)
    plt.close(fig)


def format_econf(econf: str) -> str:
    """Return an electron configuration suitable for printing."""
    for noble_atnum in NOBLE_ATNUMS[::-1]:
        econf_noble = klechkowski(noble_atnum)
        if econf_noble in econf and econf != econf_noble:
            return econf.replace(econf_noble, f"[{SYMBOLS[noble_atnum]}]")
    return econf


ANGMOM_CHARACTERS = 'spdfghiklmnoqrtuvwxyzabce'


def char2angqn(char: str) -> int:
    """Return the angular momentum quantum number corresponding to a character."""
    return ANGMOM_CHARACTERS.index(char.lower())


def interpret_econf(econf: str) -> List[np.ndarray]:
    """Translate econf strings into occupation numbers per angular momentum.

    Parameters
    ----------
    econf
        An electronic configuration string, e.g '1s2 2s2 2p3.5' for oxygen with
        7.5 electons.

    Returns
    -------
    occups
        A list of arrays, one array per angular momentum (in the order s, p, d,
        ...). Each array contains the electron occupation numbers for the radial
        orbitals for a given angular momentum. The maximum occupation number is
        ``2 * (2 * angqn + 1)``. The radial orbitals for a given angular
        momentum quantum number are assumed to be ordered from low to high. The
        occupation number arrays are truncated after the highest occupied
        orbital, i.e. remaining zero occupation numbers are not included.

    """
    occups: List[List[float]] = []
    for key in econf.split():
        occup = float(key[2:])
        if occup <= 0:
            raise ValueError("Occuptions in the electronic configuration must "
                             "be strictly positive.")
        priqn = int(key[0])
        angqn = char2angqn(key[1])
        if occup > 2 * (2 * angqn + 1):
            raise ValueError("Occuptions in the electronic configuration must "
                             "not exceed 2 * (2 * angqn + 1).")
        # Add more angular momenta if needed.
        while len(occups) < angqn + 1:
            occups.append([])
        # Add more energy levels
        i = priqn - angqn - 1
        while len(occups[angqn]) < i + 1:
            occups[angqn].append(0.0)
        occups[angqn][i] = occup
    return [np.array(angqn_occups) for angqn_occups in occups]


def klechkowski(nelec: float) -> str:
    """Return the atomic electron configuration by following the Klechkowski rule."""
    priqn_plus_angqn = 1
    words = []
    while nelec > 0:
        # start with highest possible angqn for given priqn+angqn
        angqn = (priqn_plus_angqn - 1) // 2
        priqn = priqn_plus_angqn - angqn
        while nelec > 0 and angqn >= 0:
            nelec_orb = min(nelec, 2 * (2 * angqn + 1))
            nelec -= nelec_orb
            words.append("{}{}{}".format(priqn, ANGMOM_CHARACTERS[angqn], nelec_orb))
            angqn -= 1
            priqn += 1
        priqn_plus_angqn += 1
    return " ".join(words)


SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
    "Mc", "Lv", "Ts", "Og"
]


NOBLE_ATNUMS = [2, 10, 18, 36, 54, 86]


if __name__ == '__main__':
    main()

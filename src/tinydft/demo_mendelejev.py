#!/usr/bin/env python3
# Tiny DFT is a minimalistic atomic DFT implementation.
# Copyright (C) 2024 The Tiny DFT Development Team
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


import matplotlib.pyplot as plt
import numpy as np

from .atom import NOBLE_ATNUMS, SYMBOLS, format_econf, interpret_econf, klechkowski
from .basis import Basis
from .dft import build_rho, scf_atom
from .grid import setup_grid


def main():
    """Compute atomic DFT results for a series of elements."""
    for atnum in range(1, 19):
        atcharge = 0
        nelec = atnum - atcharge
        econf = klechkowski(nelec)
        fn_out = "rho_z{:03d}_{}.png".format(atnum, "_".join(econf.split()))
        plot_atom(atnum, econf, fn_out)


def plot_atom(atnum: float, econf: str, fn_out: str):
    """Run an atomic DFT calculation and make a nice plot of the density.

    Parameters
    ----------
    atnum
        Atomic number.
    econf
        Electronic configuration.
    fn_out
        Path of the density plot.

    """
    occups = interpret_econf(econf)
    nelec = sum(occup.sum() for occup in occups)
    print(f"Nuclear charge                    {atnum:8.1f}")
    print(f"Number of electrons               {nelec:8.1f}")
    print(f"Electronic configuration          {econf!s:s}")

    grid = setup_grid()
    basis = Basis(grid)
    # Compute the overlap matrix, just to check the numerics of the basis.
    evals_olp = np.linalg.eigvalsh(basis.olp)
    print(f"Number of radial grid points      {len(grid.points):8d}")
    print(f"Number of basis functions         {basis.nbasis:8d}")
    print(f"Condition number of the overlap   {evals_olp[-1] / evals_olp[0]:8.1e}")

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
        evals[occups[angqn] > 0].max() for angqn, (evals, _evecs) in enumerate(eps_orbs_u)
    )

    # Make a nice plot of the densities.
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.text(1, 1, format_econf(econf), ha="right", va="top", transform=ax.transAxes)
    ax.text(
        1,
        0.85,
        (
            f"Z={atnum}"
            "\n"
            rf"$E_{{\mathrm{{KS}}}}$ = {energies[0]:.2f} $\mathrm{{E_h}}$"
            "\n"
            rf"-$\epsilon_N$ = {-eps_homo:.2f} $\mathrm{{E_h}}$"
        ),
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(1, 0.5, SYMBOLS[atnum], ha="right", va="top", transform=ax.transAxes, fontsize=28)
    ax.semilogy(grid.points, rho_core / 2, "k-", alpha=0.25)
    ax.semilogy(grid.points, rho_beta, "-", color="C0", lw=3, alpha=1.0)
    ax.semilogy(grid.points, rho_alpha, "-", color="C1", alpha=1.0)
    ax.semilogy(grid.points, rho, ":", color="C7")
    ax.set_xlabel(r"Distance [$\mathrm{a_0}$]")
    ax.set_ylabel(r"Density [$1/\mathrm{a_0}^3$]")
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlim(0, 6)
    fig.savefig(fn_out)
    plt.close(fig)


if __name__ == "__main__":
    main()

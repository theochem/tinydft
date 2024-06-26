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
"""DFT main routines."""

import autograd.numpy as agnp
import numpy as np
from autograd import elementwise_grad
from numpy.testing import assert_allclose
from scipy.linalg import eigh

from .basis import Basis
from .grid import TransformedGrid

__all__ = ("scf_atom", "solve_poisson", "xcfunctional")


def scf_atom(
    atnum: float,
    occups: list[np.ndarray],
    grid: TransformedGrid,
    basis: Basis,
    nscf: int = 25,
    mixing: float = 0.5,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Perform a self-consistent field atomic calculation.

    Parameters
    ----------
    atnum
        The nuclear charge.
    occups
        Occupation numbers, see interpret_econf.
    grid
        A radial grid.
    basis
        The radial orbital basis.
    nscf
        The number of SCF cycles.
    mixing
        The SCF mixing parameter. 1 means the old Fock operator is not mixed in.

    Returns
    -------
    energies
        The atomic energy and its contributions.
    eps_orbs_u
        A list of tuples of (orbital energy, orbital coefficients). One tuple
        for each angular momentum quantum number. The orbital coefficients
        represent the radial solutions U = R/r.

    """
    # Fock operators from previous iteration, used for mixing.
    focks_old = []
    # Volume element in spherical coordinates
    vol = 4 * np.pi * grid.points**2

    nelec = np.concatenate(occups).sum()
    maxangqn = len(occups) - 1
    print(f"Occupation numbers per ang. mom.  {occups}")
    print(f"Number of electrons               {nelec:8.1f}")
    print(f"Maximum ang. mol. quantum number  {maxangqn:8d}")
    print()
    print(f"Number of SCF iterations          {nscf:8d}")
    print(f"Mixing parameter                  {mixing:8.3f}")
    print()

    def excfunction(rho, np):
        """Compute the exchange(-correlation) energy density."""
        clda = (3 / 4) * (3.0 / np.pi) ** (1 / 3)
        return -clda * rho ** (4 / 3)

    print(
        " It           Total         Rad Kin         Ang Kin         Hartree "
        "             XC             Ext"
    )
    print(
        "=== =============== =============== =============== =============== "
        "=============== ==============="
    )

    # SCF cycle
    # For the first iteration, the density is set to zero to obtain the core guess.
    rho = np.zeros(len(grid.points))
    vhartree = np.zeros(len(grid.points))
    vxc = np.zeros(len(grid.points))
    for iscf in range(nscf):
        # A) Solve for each angular momentum the radial Schrodinger equation.
        # orbitals energy and radial orbitals: U = R/r
        eps_orbs_u = []
        energy_ext = 0.0
        energy_kin_rad = 0.0
        energy_kin_ang = 0.0
        # Hartree and XC potential are the same for all angular momenta.
        jxc = basis.pot(vhartree + vxc)
        for angqn in range(maxangqn + 1):
            # The new fock matrix.
            fock = basis.kin_rad + atnum * basis.ext + jxc
            if angqn > 0:
                angmom_factor = (angqn * (angqn + 1)) / 2
                fock += basis.kin_ang * angmom_factor
            # Mix with the old fock matrix, if available.
            if iscf == 0:
                fock_mix = fock
                focks_old.append(fock)
            else:
                fock_mix = mixing * fock + (1 - mixing) * focks_old[angqn]
                focks_old[angqn] = fock_mix
            # Solve for the occupied orbitals.
            evals, evecs = eigh(fock_mix, basis.olp, subset_by_index=(0, len(occups[angqn]) - 1))
            eps_orbs_u.append((evals, evecs))
            # Compute the kinetic energy contributions using the orbitals.
            energy_kin_rad += np.einsum("i,ji,jk,ki", occups[angqn], evecs, basis.kin_rad, evecs)
            energy_ext += atnum * np.einsum("i,ji,jk,ki", occups[angqn], evecs, basis.ext, evecs)
            if angqn > 0:
                energy_kin_ang += (
                    np.einsum("i,ji,jk,ki", occups[angqn], evecs, basis.kin_ang, evecs)
                    * angmom_factor
                )

        # B) Build the density and derived quantities.
        # Compute the total density.
        rho = build_rho(occups, eps_orbs_u, grid, basis)
        # Check the total number of electrons.
        assert_allclose(grid.integrate(rho * vol), nelec, atol=1e-10, rtol=0)
        # Solve the Poisson problem for the new density.
        vhartree = solve_poisson(grid, rho)
        energy_hartree = 0.5 * grid.integrate(vhartree * rho * vol)
        # Compute the exchange-correlation potential and energy density.
        exc, vxc = xcfunctional(rho, excfunction)
        energy_xc = grid.integrate(exc * vol)
        # Compute the total energy.
        energy = energy_kin_rad + energy_kin_ang + energy_hartree + energy_xc + energy_ext
        print(
            f"{iscf:3d} {energy:15.6f} {energy_kin_rad:15.6f} {energy_kin_ang:15.6f} "
            f"{energy_hartree:15.6f} {energy_xc:15.6f} {energy_ext:15.6f}"
        )

    # Assemble return values
    energies = np.array(
        [energy, energy_kin_rad, energy_kin_ang, energy_hartree, energy_xc, energy_ext]
    )
    return energies, eps_orbs_u


def build_rho(
    occups: list[np.ndarray],
    eps_orbs_u: list[tuple[np.ndarray, np.ndarray]],
    grid: TransformedGrid,
    basis: Basis,
) -> np.ndarray:
    """Construct the radial electron density on a grid.

    Parameters
    ----------
    occups
        Occupation numbers, see interpret_econf.
    eps_orbs_u
        A list of tuples of (orbital energy, orbital coefficients). One tuple
        for each angular momentum quantum number. The orbital coefficients
        represent the radial solutions U = R/r.
    grid
        The grid on which the densities are evaluated.
    basis
        The basis set.

    Returns
    -------
    rho
        The electron density on the grid.
    """
    rho = 0.0
    for angqn, (_evals, evecs) in enumerate(eps_orbs_u):
        if angqn >= len(occups):
            break
        orbs_grid_u = np.dot(evecs.T, basis.fnvals)
        orbs_grid_r = orbs_grid_u / grid.points / np.sqrt(4 * np.pi)
        angqn_occups = occups[angqn]
        rho += np.dot(angqn_occups, orbs_grid_r[: len(angqn_occups)] ** 2)
    return rho


def solve_poisson(grid: TransformedGrid, rho: np.ndarray) -> np.ndarray:
    """Solve the radial poisson equation for a spherically symmetric density."""
    norm = grid.integrate(4 * np.pi * grid.points**2 * rho)
    int1 = grid.antiderivative(grid.points * rho)
    int1 -= int1[0]
    int2 = grid.antiderivative(int1)
    int2 -= int2[0]
    pot = -(4 * np.pi) * int2
    alpha = (norm - pot[-1]) / grid.points[-1]
    pot += alpha * grid.points
    pot /= grid.points
    return pot


def xcfunctional(rho: np.ndarray, excfunction) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exchange-(correlation) energy density and potential."""
    exc = excfunction(rho, np)
    vxc = elementwise_grad(excfunction)(rho, agnp)
    return exc, vxc

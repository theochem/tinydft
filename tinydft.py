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
"""Tiny DFT is a minimalistic atomic DFT implementation.

It only supports closed-shell calculations with pure local (not even semi-local)
functionals. One has to fix the occupations numbers of the atomic orbitals a
priori.

Atomic units are used throughout.
"""

from typing import List, Tuple

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from tinybasis import Basis
from tinygrid import TransformedGrid


# np.set_printoptions(suppress=True, precision=4, linewidth=500)


__all__ = [
    'main', 'scf_atom', 'setup_grid', 'solve_poisson', 'xcfunctional', 'char2angqn',
    'interpret_econf', 'klechkowski']


def main(atnum: float, atcharge: float):
    """Run an atomic DFT calculation.

    Parameters
    ----------
    atnum
        Atomic number.
    atcharge
        The net charge.

    """
    econf = klechkowski(atnum - atcharge)
    occups = interpret_econf(econf)
    print("Nuclear charge                    {:8.1f}".format(atnum))
    print("Electronic configuration          {:s}".format(str(econf)))

    grid = setup_grid()
    basis = Basis(grid)
    # Compute the overlap matrix.
    evals_olp = np.linalg.eigvalsh(basis.olp)
    print("Number of radial grid points      {:8d}".format(len(grid.points)))
    print("Number of basis functions         {:8d}".format(basis.nbasis))
    print("Condition number of the overlap   {:8.1e}".format(evals_olp[-1] / evals_olp[0]))

    energies, rho = scf_atom(atnum, occups, grid, basis)

    plt.clf()
    plt.title("Z={:d} Energy={:.5f}".format(atnum, energies[0]))
    plt.semilogy(grid.points, rho)
    plt.xlabel("Distance from nucleus")
    plt.ylabel("Density")
    plt.ylim(1e-4, rho.max() * 2)
    plt.xlim(0, 5)
    plt.savefig("rho_z{:03d}_{}.png".format(atnum, '_'.join(econf.split())))


# pylint: disable=too-many-statements
def scf_atom(atnum: float, occups: List[List[float]], grid: TransformedGrid,
             basis: Basis, nscf: int = 25, mixing: float = 0.5) \
        -> Tuple[np.ndarray, np.ndarray]:
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
    rho
        The electron density on the grid points.

    """
    # Fock operators from previous iteration, used for mixing.
    focks_old = []
    # Volume element in spherical coordinates
    vol = 4 * np.pi * grid.points**2

    nelec = np.concatenate(occups).sum()
    maxangqn = len(occups) - 1
    print("Occupation numbers per ang. mom.  {:}".format(occups))
    print("Number of electrons               {:8.1f}".format(nelec))
    print("Maximum ang. mol. quantum number  {:8d}".format(maxangqn))
    print()
    print("Number of SCF iterations          {:8d}".format(nscf))
    print("Mixing parameter                  {:8.3f}".format(mixing))
    print()

    # pylint: disable=redefined-outer-name
    def excfunction(rho, np):
        """Compute the exchange(-correlation) energy density."""
        clda = (3 / 4) * (3.0 / np.pi)**(1 / 3)
        return -clda * rho**(4 / 3)

    print(" It           Total         Rad Kin         Ang Kin         Hartree "
          "             XC             Ext")
    print("=== =============== =============== =============== =============== "
          "=============== ===============")

    # SCF cycle
    # For the first iteration, the density is set to zero to obtain the core guess.
    rho = np.zeros(len(grid.points))
    vhartree = np.zeros(len(grid.points))
    vxc = np.zeros(len(grid.points))
    for iscf in range(nscf):
        # A) Solve for each angular momentum the radial Schrodinger equation.
        eps_orbs_u = {}  # orbitals energy and radial orbitals: U = R/r
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
            # Solve for the orbitals.
            evals, evecs = eigh(fock_mix, basis.olp, eigvals=(0, len(occups[angqn]) - 1))
            eps_orbs_u[angqn] = evals, evecs
            # Compute the kinetic energy contributions using the orbitals.
            energy_kin_rad += np.einsum(
                'i,ji,jk,ki', occups[angqn], evecs, basis.kin_rad, evecs)
            energy_ext += atnum * np.einsum(
                'i,ji,jk,ki', occups[angqn], evecs, basis.ext, evecs)
            if angqn > 0:
                energy_kin_ang += np.einsum(
                    'i,ji,jk,ki',
                    occups[angqn], evecs, basis.kin_ang, evecs) * angmom_factor

        # B) Build the density and derived quantities.
        # Compute the total density.
        rho = 0.0
        for angqn in range(maxangqn + 1):
            orbs_grid_u = np.dot(eps_orbs_u[angqn][1].T, basis.fnvals)
            orbs_grid_r = orbs_grid_u / grid.points / np.sqrt(4 * np.pi)
            rho += np.dot(occups[angqn], orbs_grid_r**2)
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
        print("{:3d} {:15.6f} {:15.6f} {:15.6f} {:15.6f} {:15.6f} {:15.6f}".format(
            iscf, energy, energy_kin_rad, energy_kin_ang, energy_hartree, energy_xc,
            energy_ext))

    energies = np.array([energy, energy_kin_rad, energy_kin_ang, energy_hartree,
                         energy_xc, energy_ext])
    return energies, rho


def setup_grid(npoint: int = 256) -> TransformedGrid:
    """Create a suitable grid for integration and differentiation."""
    # pylint: disable=redefined-outer-name
    def transform(x: np.ndarray, np) -> np.ndarray:
        """Transform from [-1, 1] to [0, big_radius]."""
        left = 1e-3
        right = 1e4
        alpha = np.log(right / left)
        return left * (np.exp(alpha * (1 + x) / 2) - 1)
    return TransformedGrid(transform, npoint)


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


def xcfunctional(rho: np.ndarray, excfunction) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the exchange-(correlation) energy density and potential."""
    from autograd import elementwise_grad
    import autograd.numpy as agnp
    exc = excfunction(rho, np)
    # pylint: disable=no-value-for-parameter
    vxc = elementwise_grad(excfunction)(rho, agnp)
    return exc, vxc


ANGMOM_CHARACTERS = 'spdfghiklmnoqrtuvwxyzabce'


def char2angqn(char: str) -> int:
    """Return the angular momentum quantum number corresponding to a character."""
    return ANGMOM_CHARACTERS.index(char.lower())


def interpret_econf(econf: str) -> List[List[float]]:
    """Translate econf strings into occupation numbers per angular momentum.

    Parameters
    ----------
    econf
        An electronic configuration string, e.g '1s2 2s2 2p3.5' for oxygen with
        7.5 electons.

    Returns
    -------
    occups
        A list of lists, one per angular momentum, with the electron occupation
        for each orbitals. The maximum occupation is 2 * (2 * angqn + 1).

    """
    occups: List[List[float]] = []
    for key in econf.split():
        occup = float(key[2:])
        if occup <= 0:
            raise TypeError("Occuptions in the electronic configuration must "
                            "be strictly positive.")
        priqn = int(key[0])
        angqn = char2angqn(key[1])
        while len(occups) < angqn + 1:
            occups.append([])
        i = priqn - angqn - 1
        while len(occups[angqn]) < i + 1:
            occups[angqn].append(0.0)
        occups[angqn][i] = occup
    return occups


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


if __name__ == '__main__':
    main(23, 0)

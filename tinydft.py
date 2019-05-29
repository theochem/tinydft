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

from __future__ import print_function, division  # Py 2.7 compatibility

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from tinybasis import Basis
from tinygrid import TransformedGrid


# np.set_printoptions(suppress=True, precision=4, linewidth=500)


__all__ = [
    'main', 'scf_atom', 'setup_grid', 'solve_poisson', 'xcfunctional', 'char2l',
    'interpret_econf', 'klechkowski']


def main(z, q):
    """Run an atomic DFT calculation.

    Parameters
    ----------
    z
        Atomic number.
    q
        The net charge.

    """
    econf = klechkowski(z - q)
    occups = interpret_econf(econf)
    print("Nuclear charge                    {:8d}".format(z))
    print("Electronic configuration          {:s}".format(str(econf)))

    grid = setup_grid()
    basis = Basis(grid)
    # Compute the overlap matrix.
    evals_olp = np.linalg.eigvalsh(basis.olp)
    print("Number of radial grid points      {:8d}".format(len(grid.points)))
    print("Number of basis functions         {:8d}".format(basis.nbasis))
    print("Condition number of the overlap   {:8.1e}".format(evals_olp[-1] / evals_olp[0]))

    energies, rho = scf_atom(z, occups, grid, basis)

    plt.clf()
    plt.title("Z={:d} Energy={:.5f}".format(z, energies[0]))
    plt.semilogy(grid.points, rho)
    plt.xlabel("Distance from nucleus")
    plt.ylabel("Density")
    plt.ylim(1e-4, rho.max() * 2)
    plt.xlim(0, 5)
    plt.savefig("rho_z{:03d}_{}.png".format(z, '_'.join(econf.split())))


def scf_atom(z, occups, grid, basis, nscf=25, mixing=0.5):
    """Perform a self-consistent field atomic calculation.

    Parameters
    ----------
    z
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
    # Dictionary for Fock operators from previous iteration, used for mixing.
    ops_fock_old = {}

    nelec = np.concatenate(occups).sum()
    maxl = len(occups) - 1
    print("Occupation numbers per ang. mom.  {:}".format(occups))
    print("Number of electrons               {:8.1f}".format(nelec))
    print("Maximum ang. mol. quantum number  {:8d}".format(maxl))
    print()
    print("Number of SCF iterations          {:8d}".format(nscf))
    print("Mixing parameter                  {:8.3f}".format(mixing))
    print()

    op_core_s = basis.kin_rad + z * basis.ext
    vol = 4 * np.pi * grid.points**2

    def excfunction(rho, np):
        """Compute the exchange(-correlation) energy density."""
        clda = (3 / 4) * (3.0 / np.pi)**(1 / 3)
        return -clda * rho**(4 / 3)

    print(" It           Total         Rad Kin         Ang Kin         Hartree "
          "             XC             Ext")
    print("=== =============== =============== =============== =============== "
          "=============== ===============")

    # SCF cycle
    for iscf in range(nscf):
        # A) Build the density and derived quantities.
        if iscf == 0:
            # In the first iteration, the density is set to zero to obtain the
            # core guess.
            rho = np.zeros(len(grid.points))
            vhartree = np.zeros(len(grid.points))
            vxc = np.zeros(len(grid.points))
            print("  0      (guess)")
        else:
            # Compute the total density.
            rho = 0.0
            for l in range(maxl + 1):
                orbs_grid_u = np.dot(eps_orbs_u[l][1].T, basis.fns)
                orbs_grid_r = orbs_grid_u / grid.points / np.sqrt(4 * np.pi)
                rho += np.dot(occups[l], orbs_grid_r**2)
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

        # B) Solve for each angular momentum the radial Schrodinger equation.
        eps_orbs_u = {}  # orbitals energy and radial orbitals: U = R/r
        energy_ext = 0.0
        energy_kin_rad = 0.0
        energy_kin_ang = 0.0
        # Hartree and XC potential are the same for all angular momenta.
        op_jxc = basis.pot(vhartree + vxc)
        for l in range(maxl + 1):
            # The new fock matrix.
            op_fock = op_core_s + op_jxc
            if l > 0:
                angmom_factor = (l * (l + 1)) / 2
                op_fock += basis.kin_ang * angmom_factor
            # Mix with the old fock matrix, if available.
            op_fock_old = ops_fock_old.get(l)
            if op_fock_old is None:
                op_fock_mix = op_fock
            else:
                op_fock_mix = mixing * op_fock + (1 - mixing) * op_fock_old
            ops_fock_old[l] = op_fock_mix
            # Solve for the orbitals.
            evals, evecs = eigh(op_fock_mix, basis.olp, eigvals=(0, len(occups[l]) - 1))
            eps_orbs_u[l] = evals, evecs
            # Compute the kinetic energy contributions using the orbitals.
            energy_kin_rad += np.einsum('i,ji,jk,ki', occups[l], evecs, basis.kin_rad, evecs)
            energy_ext += z * np.einsum('i,ji,jk,ki', occups[l], evecs, basis.ext, evecs)
            if l > 0:
                energy_kin_ang += (np.einsum('i,ji,jk,ki', occups[l], evecs, basis.kin_ang, evecs)
                                   * angmom_factor)

    # Plot the electron density.
    energies = np.array([energy, energy_kin_rad, energy_kin_ang, energy_hartree,
                         energy_xc, energy_ext])
    return energies, rho


def setup_grid(npoint=256):
    """Create a suitable grid for integration and differentiation."""
    def tf(t, np):
        """Transform from [-1, 1] to [0, big_radius]."""
        u = (1 + t) / 2
        left = 1e-3
        right = 1e4
        alpha = np.log(right / left)
        return left * (np.exp(alpha * u) - 1)
    return TransformedGrid(tf, npoint)


def solve_poisson(grid, rho):
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


def xcfunctional(rho, excfunction):
    """Compute the exchange-(correlation) energy density and potential."""
    from autograd import elementwise_grad
    import autograd.numpy as np
    exc = excfunction(rho, np)
    vxc = elementwise_grad(excfunction)(rho, np)
    return exc, vxc


ANGMOM_CHARACTERS = 'spdfghiklmnoqrtuvwxyzabce'


def char2l(char):
    """Return the angular momentum quantum number corresponding to a character."""
    return ANGMOM_CHARACTERS.index(char.lower())


def interpret_econf(econf):
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
        for each orbitals. The maximum occupation is 2 * (2 * l + 1).

    """
    occups = []
    for key in econf.split():
        occup = float(key[2:])
        if occup <= 0:
            raise TypeError("Occuptions in the electronic configuration must "
                            "be strictly positive.")
        n = int(key[0])
        l = char2l(key[1])
        while len(occups) < l + 1:
            occups.append([])
        i = n - l - 1
        while len(occups[l]) < i + 1:
            occups[l].append([])
        occups[l][i] = occup
    return occups


def klechkowski(nelec):
    """Return the atomic electron configuration by following the Klechkowski rule."""
    n_plus_l = 1
    words = []
    while nelec > 0:
        # start with highest possible l for given n+l
        l = (n_plus_l - 1) // 2
        n = n_plus_l - l
        while nelec > 0 and l >= 0:
            nelec_orb = min(nelec, 2 * (2 * l + 1))
            nelec -= nelec_orb
            words.append("{}{}{}".format(n, ANGMOM_CHARACTERS[l], nelec_orb))
            l -= 1
            n += 1
        n_plus_l += 1
    return " ".join(words)


if __name__ == '__main__':
    main(23, 0)

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

from tinygrid import TransformedGrid


# np.set_printoptions(suppress=True, precision=4, linewidth=500)


__all__ = [
    'main', 'setup_grid', 'setup_obasis', 'compute_overlap_operator',
    'compute_radial_kinetic_operator', 'compute_angular_kinetic_operator',
    'compute_potential_operator', 'solve_poisson', 'xcfunctional', 'char2l',
    'interpret_econf']


def main(z, econf, nscf=25, mixing=0.5):
    """Run an atomic DFT calculation.

    Parameters
    ----------
    z
        Atomic number.
    econf
        An electronic configuration string, e.g '1s2 2s2 2p3.5' for oxygen with
        7.5 electons.
    nscf
        The number of SCF cycles.
    mixing
        The coefficient for mixing old and new fock matrices. 1.0 means no
        mixing.

    Returns
    -------
    energy
        The atomic energy.

    """
    occups = interpret_econf(econf)
    nelec = np.concatenate(occups).sum()
    maxl = len(occups) - 1
    print("Electronic configuration          {:s}".format(str(econf)))
    print("Occupation numbers per ang. mom.  {:}".format(occups))
    print("Nuclear charge                    {:8d}".format(z))
    print("Number of electrons               {:8.1f}".format(nelec))
    print("Maximum ang. mol. quantum number  {:8d}".format(maxl))
    print()

    grid = setup_grid()
    vol = 4 * np.pi * grid.points**2
    npoint = len(grid.points)
    obasis = setup_obasis(grid)
    # Compute the overlap matrix.
    overlap = compute_overlap_operator(grid, obasis)
    evals_olp = np.linalg.eigvalsh(overlap)
    print("Number of radial grid points      {:8d}".format(len(grid.points)))
    print("Number of basis functions         {:8d}".format(obasis.shape[0]))
    print("Condition number of the overlap   {:8.1e}".format(evals_olp[0] / evals_olp[-1]))
    print()

    def excfunction(rho, np):
        """Compute the exchange(-correlation) energy density."""
        clda = (3 / 4) * (3.0 / np.pi)**(1 / 3)
        return -clda * rho**(4 / 3)

    print("Pre-computing some integrals ...")
    print()
    # Radial kinetic energy.
    op_kin_rad = compute_radial_kinetic_operator(grid, obasis)
    # Interaction with the nuclear potential.
    vext = -z / grid.points
    op_ext = compute_potential_operator(grid, obasis, vext)
    # The core Hamiltonian for l = 0 (s).
    op_core_s = op_kin_rad + op_ext
    # angular kinetic energy operators.
    ops_kin_ang = {}
    for l in range(1, maxl + 1):
        op_kin_ang = compute_angular_kinetic_operator(grid, obasis, l)
        ops_kin_ang[l] = op_kin_ang

    # Dictionary for Fock operators from previous iteration, used for mixing.
    ops_fock_old = {}

    print("Number of SCF iterations          {:8d}".format(nscf))
    print("Mixing parameter                  {:8.3f}".format(mixing))
    print()

    print(" It        Total      Rad Kin      Ang Kin      Hartree           XC          Ext")
    print("=== ============ ============ ============ ============ ============ ============")

    # SCF cycle
    for iscf in range(nscf):
        # A) Build the density and derived quantities.
        if iscf == 0:
            # In the first iteration, the density is set to zero to obtain the
            # core guess.
            rho = np.zeros(npoint)
            vhartree = np.zeros(npoint)
            vxc = np.zeros(npoint)
            print("  0      (guess)")
        else:
            # Compute the total density.
            rho = 0.0
            for l in range(maxl + 1):
                for i, occup in enumerate(occups[l]):
                    # orbital on grid (U)
                    orb_grid_u = np.dot(eps_orbs_u[l][1][:, i], obasis)
                    # orbital on grid (R) with normalization of spherical harmonic.
                    orb_grid_r = orb_grid_u / grid.points / np.sqrt(4 * np.pi)
                    rho += occup * (orb_grid_r)**2
            # Check the total number of electrons.
            assert_allclose(grid.integrate(rho * vol), nelec)
            # Solve the Poisson problem for the new density.
            vhartree = solve_poisson(grid, rho)
            energy_hartree = 0.5 * grid.integrate(vhartree * rho * vol)
            # Compute the exchange-correlation potential and energy density.
            exc, vxc = xcfunctional(rho, excfunction)
            energy_xc = grid.integrate(exc * vol)
            # Compute the interaction with the external field.
            energy_ext = grid.integrate(vext * rho * vol)
            # Compute the total energy.
            energy = energy_kin_rad + energy_kin_ang + energy_hartree + energy_xc + energy_ext
            print("{:3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}".format(
                iscf, energy, energy_kin_rad, energy_kin_ang, energy_hartree, energy_xc,
                energy_ext))

        # B) Solve for each angular momentum the radial Schrodinger equation.
        eps_orbs_u = {}  # orbitals energy and radial orbitals: U = R/r
        energy_kin_rad = 0.0
        energy_kin_ang = 0.0
        # Hartree and XC potential are the same for all angular momenta.
        op_jxc = compute_potential_operator(grid, obasis, vhartree + vxc)
        for l in range(maxl + 1):
            # The new fock matrix.
            op_fock = op_core_s + op_jxc
            if l > 0:
                op_fock += ops_kin_ang[l]
            # Mix with the old fock matrix, if available.
            op_fock_old = ops_fock_old.get(l)
            if op_fock_old is None:
                op_fock_mix = op_fock
            else:
                op_fock_mix = mixing * op_fock + (1 - mixing) * op_fock_old
            ops_fock_old[l] = op_fock_mix
            # Solve for the orbitals.
            evals, evecs = eigh(op_fock_mix, overlap)
            eps_orbs_u[l] = evals, evecs
            # Compute the kinetic energy contributions using the orbitals.
            for i, occup in enumerate(occups[l]):
                orb_u = evecs[:, i]
                energy_kin_rad += occup * np.einsum('i,ij,j', orb_u, op_kin_rad, orb_u)
                if l > 0:
                    energy_kin_ang += occup * np.einsum('i,ij,j', orb_u, op_kin_ang, orb_u)

    # Plot the electron density.
    plt.clf()
    plt.title("Z={:d} Energy={:.5f}".format(z, energy))
    plt.semilogy(grid.points, rho)
    plt.xlabel("Distance from nucleus")
    plt.ylabel("Density")
    plt.ylim(1e-4, rho.max() * 2)
    plt.xlim(0, 5)
    plt.savefig("rho_z{:03d}_{}.png".format(z, '_'.join(econf.split())))

    return energy


def setup_grid(npoint=512):
    """Create a suitable grid for integration and differentiation."""
    def tf(t, np):
        """Transform from [-1, 1] to [0, big_radius]."""
        u = (1 + t) / 2
        # return 50 * u**2
        # return np.arctanh(u)**2
        return 1e-5 * np.exp(20 * u)
    return TransformedGrid(tf, npoint)


def setup_obasis(grid, nbasis=40):
    """Define a radial orbital basis set on a grid, for the U=R/r functions."""
    # The basis functions are simply even-tempered Gaussians.
    alphas = 10**np.linspace(-2, 4, nbasis)
    obasis = []
    for alpha in alphas:
        # Note the multiplication with the radius.
        fn = np.exp(-alpha * grid.points**2) * grid.points
        fn /= np.sqrt(grid.integrate(fn**2))
        obasis.append(fn)
    return np.array(obasis)


def compute_operator(obasis, compute):
    """Driver for operator computation.

    Parameters
    ----------
    obasis
        The orbital basis array.
    compute
        A function taking two basis functions and returning the antiderivative
        defining the operator.

    Returns
    -------
    operator
        (nbasis, nbasis) array.

    """
    nbasis = obasis.shape[0]
    operator = np.zeros((nbasis, nbasis))
    for i0, fn0 in enumerate(obasis):
        for i1, fn1 in enumerate(obasis[:i0 + 1]):
            operator[i0, i1] = compute(fn0, fn1)
            operator[i1, i0] = operator[i0, i1]
    return operator


def compute_overlap_operator(grid, obasis):
    """Return the overlap operator."""
    return compute_potential_operator(grid, obasis, 1)


def compute_radial_kinetic_operator(grid, obasis):
    """Return the radial kinetic energy operator."""
    return compute_operator(obasis, (
        lambda fn0, fn1: grid.integrate(-fn0 * grid.derivative(fn1, 2)) / 2
        # lambda fn0, fn1: grid.integrate(grid.derivative(fn0) * grid.derivative(fn1)) / 2
    ))


def compute_angular_kinetic_operator(grid, obasis, l):
    """Return the angular kinetic energy operator."""
    # centrifugal potential (due to angular velocity)
    v_angkin = l * (l + 1) / (2 * grid.points**2)
    return compute_potential_operator(grid, obasis, v_angkin)


def compute_potential_operator(grid, obasis, v):
    """Return an operator for the given potential v."""
    return compute_operator(obasis, (lambda fn0, fn1: grid.integrate(fn0 * fn1 * v)))


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


def char2l(char):
    """Return the angular momentum quantum number corresponding to a character."""
    return 'spdfghiklmnoqrtuvwxyzabce'.index(char.lower())


def interpret_econf(econf):
    """Translate econf strings into occupation numbers per angular momentum."""
    occups = []
    for key in econf.split():
        occup = float(key[2:])
        if occup > 0:
            n = int(key[0])
            l = char2l(key[1])
            while len(occups) < l + 1:
                occups.append([])
            i = n - l - 1
            while len(occups[l]) < i + 1:
                occups[l].append([])
            occups[l][i] = occup
    return occups


if __name__ == '__main__':
    main(23, '1s2 2s2 2p6 3s2 3p6 4s2 3d3')

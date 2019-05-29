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
"""Unit tests for Tiny DFT."""

from __future__ import print_function, division  # Py 2.7 compatibility

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre
import pytest

from tinydft import setup_grid
from tinybasis import Basis


def get_hydrogenic_solutions(grid, z, l):
    """Compute the hydrogenic orbitals on a radial grid.

    Parameters
    ----------
    grid
        The radial integration grid.
    z
        The nuclear charge.
    l
        The angular momentum quantum number.

    Returns
    -------
    psis
        List of tuples: (n, factor, psi) where n is the principal quantum
        number, factor is z**2/n**2 and psi is the radial dependence of the
        orbital U=R*r, on the grid.

    """
    psis = []
    for i in range(7 - l):
        n = i + 1 + l
        factor = z**2 / n**2

        # Compute the orbital analytically
        fac = np.math.factorial
        normalization = np.sqrt(
            (2 * z / n)**3 * fac(n - l - 1) / (2 * n * fac(n + l)))
        rho = grid.points * 2 * z / n
        poly = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
        psi = normalization * np.exp(-rho / 2) * rho**l * poly * grid.points
        psis.append((n, factor, psi))
    return psis


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_grid(z, l):
    grid = setup_grid()
    psis = get_hydrogenic_solutions(grid, z, l)
    if l > 0:
        v_angkin = l * (l + 1) / (2 * grid.points**2)
    v_ext = -z / grid.points
    for i, (n, factor, psi) in enumerate(psis):
        case = "i={} n={}".format(i, n)

        # Check the observables for the analytic solution on the grid.
        norm = grid.integrate(psi**2)
        ekin = grid.integrate(-psi * grid.derivative(grid.derivative(psi)) / 2)
        if l > 0:
            ekin += grid.integrate(psi**2 * v_angkin)
        ena = grid.integrate(psi**2 * v_ext)
        assert_allclose(norm, 1, atol=1e-14, rtol=0, err_msg=case)
        # atol is set to micro-Hartree precision
        assert_allclose(ekin, factor / 2, atol=4e-11, rtol=0, err_msg=case)
        assert_allclose(ena, -factor, atol=4e-11, rtol=0, err_msg=case)


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_op(z, l, grid_basis):
    grid, basis = grid_basis
    # Created modified copies of the operators. Do not modificy in place to
    # avoid side-effects.
    kin = basis.kin_rad.copy()
    if l > 0:
        angmom_factor = (l * (l + 1)) / 2
        kin += basis.kin_ang * angmom_factor
    ext = basis.ext * z
    evals, evecs = eigh(kin + ext, basis.olp)
    psis = get_hydrogenic_solutions(grid, z, l)
    for i, (n, factor, psi) in enumerate(psis):
        case = "i={} n={}".format(i, n)
        # Same test for the numerical solution
        norm = np.einsum('i,ij,j', evecs[:, i], basis.olp, evecs[:, i])
        ekin = np.einsum('i,ij,j', evecs[:, i], kin, evecs[:, i])
        eext = np.einsum('i,ij,j', evecs[:, i], ext, evecs[:, i])
        assert_allclose(norm, 1, atol=1e-10, rtol=0, err_msg=case)
        assert_allclose(eext, -factor, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(ekin, factor / 2, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(evals[i], -factor / 2, atol=0, rtol=1e-5, err_msg=case)


def test_integral_regression(num_regression):
    grid = setup_grid()
    basis = Basis(grid, 1e-1, 1e2, 5)
    assert basis.olp.shape == (5, 5)
    assert basis.kin_rad.shape == (5, 5)
    assert basis.kin_ang.shape == (5, 5)
    assert basis.ext.shape == (5, 5)
    num_regression.check(
        {'olp': basis.olp.ravel(),
         'kin_rad': basis.kin_rad.ravel(),
         'kin_ang': basis.kin_ang.ravel(),
         'ext': basis.ext.ravel()},
        default_tolerance={'rtol': 1e-15, 'atol': 0},
    )

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

from tinydft import *


def test_char2l():
    assert char2l('s') == 0
    assert char2l('S') == 0
    assert char2l('p') == 1
    assert char2l('P') == 1
    assert char2l('d') == 2
    assert char2l('D') == 2


def get_hydrogenic(grid, z, angmom):
    psis = []
    for i in range(7 - angmom):
        n = i + 1 + angmom
        factor = z**2 / n**2

        # Compute the orbital analytically
        fac = np.math.factorial
        normalization = np.sqrt(
            (2 * z / n)**3 * fac(n - angmom - 1) / (2 * n * fac(n + angmom)))
        rho = grid.points * 2 * z / n
        poly = eval_genlaguerre(n - angmom - 1, 2 * angmom + 1, rho)
        psi = normalization * np.exp(-rho / 2) * rho**angmom * poly * grid.points
        psis.append((n, factor, psi))
    return psis


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("angmom", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_grid(z, angmom):
    grid = setup_grid()
    psis = get_hydrogenic(grid, z, angmom)
    if angmom > 0:
        v_angkin = angmom * (angmom + 1) / (2 * grid.points**2)
    v_ext = -z / grid.points
    for i, (n, factor, psi) in enumerate(psis):
        case = "i={} n={}".format(i, n)

        # Check the observables for the analytic solution on the grid.
        norm = grid.integrate(psi**2)
        ekin = grid.integrate(-psi * grid.derivative(psi, 2) / 2)
        if angmom > 0:
            ekin += grid.integrate(psi**2 * v_angkin)
        ena = grid.integrate(psi**2 * v_ext)
        assert_allclose(norm, 1, atol=1e-14, rtol=0, err_msg=case)
        # atol is set to micro-Hartree precision
        assert_allclose(ekin, factor / 2, atol=4e-11, rtol=0, err_msg=case)
        assert_allclose(ena, -factor, atol=4e-11, rtol=0, err_msg=case)


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("angmom", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_op(z, angmom):
    grid = setup_grid()
    obasis = setup_obasis(grid)
    olp = compute_overlap_operator(grid, obasis)
    kin = compute_radial_kinetic_operator(grid, obasis)
    if angmom > 0:
        v_angkin = angmom * (angmom + 1) / (2 * grid.points**2)
        kin += compute_potential_operator(grid, obasis, v_angkin)
    v_ext = -z / grid.points
    na = compute_potential_operator(grid, obasis, v_ext)
    evals, evecs = eigh(kin + na, olp)
    psis = get_hydrogenic(grid, z, angmom)
    for i, (n, factor, psi) in enumerate(psis):
        case = "i={} n={}".format(i, n)
        # Same test for the numerical solution
        norm = np.einsum('i,ij,j', evecs[:, i], olp, evecs[:, i])
        ekin = np.einsum('i,ij,j', evecs[:, i], kin, evecs[:, i])
        ena = np.einsum('i,ij,j', evecs[:, i], na, evecs[:, i])
        assert_allclose(norm, 1, atol=1e-10, rtol=0, err_msg=case)
        assert_allclose(ena, -factor, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(ekin, factor / 2, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(evals[i], -factor / 2, atol=0, rtol=1e-5, err_msg=case)


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
def test_poisson(z):
    # Numerically solve the electrostatic potential of an S-type Slater Density
    grid = setup_grid()
    alpha = 2 * z
    aux = np.exp(-alpha * grid.points)
    rho = aux * alpha**3 / (8 * np.pi)
    assert_allclose(grid.integrate(4 * np.pi * grid.points**2 * rho), 1.0, atol=1e-11, rtol=0)
    vnum = solve_poisson(grid, rho)
    vann = (1 - aux) / grid.points - alpha * aux / 2
    assert_allclose(vnum[-1], 1.0 / grid.points[-1], atol=1e-11, rtol=0)
    assert_allclose(vnum, vann, atol=2e-5, rtol=0)


def test_interpret_econf():
    occups = interpret_econf('1s1 2s2 2p3.0')
    assert len(occups) == 2
    assert len(occups[0]) == 2
    assert len(occups[1]) == 1
    assert occups[0][0] == 1.0
    assert occups[0][1] == 2.0
    assert occups[1][0] == 3.0

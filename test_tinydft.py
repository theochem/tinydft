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


def get_hydrogenic_solutions(grid, z, l):
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


@pytest.fixture(scope="module")
def hydrogenic_ops():
    """Operators for the hydrogenic atom, computed only once for all tests."""
    grid = setup_grid()
    obasis = setup_obasis(grid)
    olp = grid.integrate(obasis, obasis)
    kin_rad = grid.integrate(obasis, -grid.derivative(grid.derivative(obasis))) / 2
    kin_ang = grid.integrate(obasis, obasis, grid.points**-2)
    na = grid.integrate(obasis, obasis, -grid.points**-1)
    return grid, obasis, olp, kin_rad, kin_ang, na


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_op(z, l, hydrogenic_ops):
    grid, obasis, olp, kin_rad, kin_ang, na = hydrogenic_ops
    # Created modified copies of the operators. Do not modificy in place to
    # avoid side-effects.
    kin = kin_rad.copy()
    if l > 0:
        angmom_factor = (l * (l + 1)) / 2
        kin += kin_ang * angmom_factor
    na = na * z
    evals, evecs = eigh(kin + na, olp)
    psis = get_hydrogenic_solutions(grid, z, l)
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
    with pytest.raises(TypeError):
        interpret_econf('1s1 2s0')
    with pytest.raises(TypeError):
        interpret_econf('1s1 2s-2')


def test_klechkowski():
    assert klechkowski(1) == '1s1'
    assert klechkowski(2) == '1s2'
    assert klechkowski(10) == '1s2 2s2 2p6'
    assert klechkowski(23) == '1s2 2s2 2p6 3s2 3p6 4s2 3d3'
    assert klechkowski(118) == ('1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 '
                                '6s2 4f14 5d10 6p6 7s2 5f14 6d10 7p6')


@pytest.mark.parametrize("z", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
def test_atom(z, num_regression, hydrogenic_ops):
    econf = klechkowski(z)
    occups = interpret_econf(econf)
    grid, obasis, overlap, op_kin_rad, op_kin_ang, op_ext = hydrogenic_ops
    energies, rho = scf_atom(occups, grid, obasis, overlap, op_kin_rad,
                             op_kin_ang, op_ext * z, nscf=100)
    num_regression.check(
        {'energies': energies},
        default_tolerance={'rtol': 1e-8, 'atol': 0},
    )

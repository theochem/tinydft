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
"""Unit tests for Tiny Grid."""

import numpy as np
from numpy.testing import assert_allclose
from numpy.polynomial.legendre import legval
import pytest
from scipy.special import eval_genlaguerre

from tinygrid import LegendreGrid, TransformedGrid


@pytest.mark.parametrize("npoint", range(10, 20))
def test_low_grid_basics(npoint):
    grid = LegendreGrid(npoint)
    assert grid.basis.shape == (npoint, npoint)
    assert grid.basis_inv.shape == (npoint, npoint)
    fn1 = np.random.uniform(0, 1, npoint)
    coeffs = grid.tocoeffs(fn1)
    fn2 = grid.tofnvals(coeffs)
    assert_allclose(fn1, fn2, atol=1e-14)
    fn3 = legval(grid.points, coeffs)
    assert_allclose(fn1, fn3, atol=1e-14)


def test_low_grid_basics_vectorized():
    shape = (5, 4, 8)
    npoint = 7
    extshape = shape + (npoint, )
    grid = LegendreGrid(npoint)
    fn1 = np.random.uniform(0, 1, extshape)
    coeffs = grid.tocoeffs(fn1)
    assert coeffs.shape == extshape
    fn2 = grid.tofnvals(coeffs)
    assert fn2.shape == extshape
    assert_allclose(fn1, fn2, atol=1e-14)
    # For legval, the first index of the coefficients array should be for
    # different polynomial orders. It returns function values at grid points,
    # for which the last index is used.
    fn3 = legval(grid.points, coeffs.transpose(3, 0, 1, 2))
    assert fn3.shape == extshape
    assert_allclose(fn1, fn3, atol=1e-14)


def test_low_grid_sin():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fnvals = np.cos(grid.points)
    fnvalsa = grid.antiderivative(fnvals)
    fnvalsa -= fnvalsa[npoint // 2]
    fnvalsd = grid.derivative(fnvals)
    assert_allclose(grid.integrate(fnvals), np.sin(1) - np.sin(-1), atol=1e-14, rtol=0)
    assert_allclose(fnvalsa, np.sin(grid.points), atol=1e-14, rtol=0)
    assert_allclose(fnvalsd, -np.sin(grid.points), atol=1e-10, rtol=0)


def test_low_grid_sin_vectorized():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fnsvals = np.cos(np.outer([1.0, 0.5, 0.2], grid.points))
    fnsvalsa = grid.antiderivative(fnsvals)
    fnsvalsa -= fnsvalsa[:, npoint // 2, np.newaxis]
    fnsvalsd = grid.derivative(fnsvals)
    integrals = [
        np.sin(1) - np.sin(-1),
        2 * np.sin(0.5) - 2 * np.sin(-0.5),
        5 * np.sin(0.2) - 5 * np.sin(-0.2),
    ]
    assert_allclose(grid.integrate(fnsvals), integrals, atol=1e-14, rtol=0)
    antiderivatives = [
        np.sin(grid.points),
        2 * np.sin(0.5 * grid.points),
        5 * np.sin(0.2 * grid.points),
    ]
    assert_allclose(fnsvalsa, antiderivatives, atol=1e-14, rtol=0)
    derivatives = [
        -np.sin(grid.points),
        -0.5 * np.sin(0.5 * grid.points),
        -0.2 * np.sin(0.2 * grid.points),
    ]
    assert_allclose(fnsvalsd, derivatives, atol=1e-10, rtol=0)
    fns_other = np.cos(np.outer([1.1, 1.2, 0.8], grid.points))
    integrals2 = np.array([
        [grid.integrate(fn1 * fn2) for fn2 in fns_other]
        for fn1 in fnsvals
    ])
    assert_allclose(grid.integrate(fnsvals, fns_other), integrals2, atol=1e-14, rtol=0)


def test_low_grid_exp():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fnvals = np.exp(grid.points)
    fnvalsa = grid.antiderivative(fnvals)
    fnvalsa += 1 - fnvalsa[npoint // 2]
    fnvalsd = grid.derivative(fnvals)
    assert_allclose(grid.integrate(fnvals), np.exp(1) - np.exp(-1), atol=1e-14, rtol=0)
    assert_allclose(fnvalsa, fnvals, atol=1e-14, rtol=0)
    assert_allclose(fnvalsd, fnvals, atol=1e-10, rtol=0)


def test_tf_grid_exp():
    # pylint: disable=redefined-outer-name
    def transform(x, np):
        return 10 * np.arctanh((1 + x) / 2)**2
    grid = TransformedGrid(transform, 201)
    fnvals = np.exp(-grid.points)
    fnvalsa = grid.antiderivative(fnvals)
    fnvalsa += -1 - fnvalsa[0]
    fnvalsd = grid.derivative(fnvals)
    assert_allclose(grid.integrate(fnvals), 1.0, atol=1e-13, rtol=0)
    assert_allclose(fnvalsa, -fnvals, atol=1e-7, rtol=0)
    assert_allclose(fnvalsd, -fnvals, atol=1e-7, rtol=0)


def test_tf_grid_exp_vectorized():
    # pylint: disable=redefined-outer-name
    def transform(x, np):
        return 15 * np.arctanh((1 + x) / 2)**2
    grid = TransformedGrid(transform, 201)
    exponents = np.array([1.0, 0.5, 2.0])
    fnsvals = np.exp(-np.outer(exponents, grid.points))
    fnsvalsa = grid.antiderivative(fnsvals)
    fnsvalsa += (-1 / exponents - fnsvalsa[:, 0])[:, np.newaxis]
    fnsvalsd = grid.derivative(fnsvals)
    assert_allclose(grid.integrate(fnsvals), 1 / exponents, atol=1e-13, rtol=0)
    assert_allclose(fnsvalsa, -fnsvals / exponents[:, np.newaxis], atol=1e-7, rtol=0)
    assert_allclose(fnsvalsd, -fnsvals * exponents[:, np.newaxis], atol=1e-7, rtol=0)
    fns_other = np.exp(-np.outer([1.1, 1.2, 0.8], grid.points))
    integrals2 = np.array([
        [grid.integrate(fn1 * fn2) for fn2 in fns_other]
        for fn1 in fnsvals
    ])
    assert_allclose(grid.integrate(fnsvals, fns_other), integrals2, atol=1e-14, rtol=0)


def get_hydrogenic_solutions(grid, atnum, angqn):
    """Compute the hydrogenic orbitals on a radial grid.

    Parameters
    ----------
    grid
        The radial integration grid.
    atnum
        The nuclear charge.
    angqn
        The angular momentum quantum number.

    Returns
    -------
    psis
        List of tuples: (priqn, factor, psi) where priqn is the principal quantum
        number, factor is atnum**2/priqn**2 and psi is the radial dependence of the
        orbital U=R*r, on the grid.

    """
    psis = []
    for i in range(7 - angqn):
        priqn = i + 1 + angqn
        factor = atnum**2 / priqn**2

        # Compute the orbital analytically
        fac = np.math.factorial
        normalization = np.sqrt(
            (2 * atnum / priqn)**3 * fac(priqn - angqn - 1) / (2 * priqn * fac(priqn + angqn)))
        rho = grid.points * 2 * atnum / priqn
        poly = eval_genlaguerre(priqn - angqn - 1, 2 * angqn + 1, rho)
        psi = normalization * np.exp(-rho / 2) * rho**angqn * poly * grid.points
        psis.append((priqn, factor, psi))
    return psis


@pytest.mark.parametrize("atnum", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("angqn", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_grid(atnum, angqn, grid_basis):
    grid = grid_basis[0]
    psis = get_hydrogenic_solutions(grid, atnum, angqn)
    if angqn > 0:
        v_angkin = angqn * (angqn + 1) / (2 * grid.points**2)
    v_ext = -atnum / grid.points
    for i, (priqn, factor, psi) in enumerate(psis):
        case = "i={} priqn={}".format(i, priqn)

        # Check the observables for the analytic solution on the grid.
        norm = grid.integrate(psi**2)
        ekin = grid.integrate(-psi * grid.derivative(grid.derivative(psi)) / 2)
        if angqn > 0:
            ekin += grid.integrate(psi**2 * v_angkin)
        eext = grid.integrate(psi**2 * v_ext)
        assert_allclose(norm, 1, atol=1e-14, rtol=0, err_msg=case)
        assert_allclose(ekin, factor / 2, atol=4e-11, rtol=0, err_msg=case)
        assert_allclose(eext, -factor, atol=4e-11, rtol=0, err_msg=case)


def test_tf_grid_hydrogen_few():
    # pylint: disable=redefined-outer-name
    def transform(x, np):
        left = 1e-2
        right = 1e3
        alpha = np.log(right / left)
        return left * (np.exp(alpha * (1 + x) / 2) - 1)
    grid = TransformedGrid(transform, 101)

    # Solutions of the radial equation (U=R/r)
    psi_1s = np.sqrt(4 * np.pi) * grid.points * np.exp(-grid.points) / np.sqrt(np.pi)
    psi_2s = np.sqrt(4 * np.pi) * grid.points * np.exp(-grid.points / 2) / \
        np.sqrt(2 * np.pi) / 4 * (2 - grid.points)

    # Check norms with vectorization
    norms = grid.integrate(np.array([psi_1s, psi_2s])**2)
    assert_allclose(norms, 1.0, atol=1e-14, rtol=0)

    # Check norms and energies
    for eps, psi in [(-0.5, psi_1s), (-0.125, psi_2s)]:
        ekin = grid.integrate(-psi * grid.derivative(grid.derivative(psi)) / 2)
        assert_allclose(ekin, -eps, atol=1e-11, rtol=0)
        epot = grid.integrate(-psi**2 / grid.points)
        assert_allclose(epot, 2 * eps, atol=1e-14, rtol=0)

    dot = grid.integrate(psi_1s * psi_2s)
    assert_allclose(dot, 0.0, atol=1e-15, rtol=0)

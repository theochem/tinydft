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

from __future__ import print_function, division  # Py 2.7 compatibility

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
    coeffs = grid._tocoeffs(fn1)
    fn2 = grid._tovalues(coeffs)
    assert_allclose(fn1, fn2, atol=1e-14)
    fn3 = legval(grid.points, coeffs)
    assert_allclose(fn1, fn3, atol=1e-14)


def test_low_grid_basics_vectorized():
    shape = (5, 4, 8)
    npoint = 7
    extshape = shape + (npoint, )
    grid = LegendreGrid(npoint)
    fn1 = np.random.uniform(0, 1, extshape)
    coeffs = grid._tocoeffs(fn1)
    assert coeffs.shape == extshape
    fn2 = grid._tovalues(coeffs)
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
    fn = np.cos(grid.points)
    fna = grid.antiderivative(fn)
    fna -= fna[npoint // 2]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), np.sin(1) - np.sin(-1), atol=1e-14, rtol=0)
    assert_allclose(fna, np.sin(grid.points), atol=1e-14, rtol=0)
    assert_allclose(fnd, -np.sin(grid.points), atol=1e-10, rtol=0)


def test_low_grid_sin_vectorized():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fns = np.cos(np.outer([1.0, 0.5, 0.2], grid.points))
    fnsa = grid.antiderivative(fns)
    fnsa -= fnsa[:, npoint // 2, np.newaxis]
    fnsd = grid.derivative(fns)
    integrals = [
        np.sin(1) - np.sin(-1),
        2 * np.sin(0.5) - 2 * np.sin(-0.5),
        5 * np.sin(0.2) - 5 * np.sin(-0.2),
    ]
    assert_allclose(grid.integrate(fns), integrals, atol=1e-14, rtol=0)
    antiderivatives = [
        np.sin(grid.points),
        2 * np.sin(0.5 * grid.points),
        5 * np.sin(0.2 * grid.points),
    ]
    assert_allclose(fnsa, antiderivatives, atol=1e-14, rtol=0)
    derivatives = [
        -np.sin(grid.points),
        -0.5 * np.sin(0.5 * grid.points),
        -0.2 * np.sin(0.2 * grid.points),
    ]
    assert_allclose(fnsd, derivatives, atol=1e-10, rtol=0)
    fns_other = np.cos(np.outer([1.1, 1.2, 0.8], grid.points))
    integrals2 = np.array([
        [grid.integrate(fn1 * fn2) for fn2 in fns_other]
        for fn1 in fns
    ])
    assert_allclose(grid.integrate(fns, fns_other), integrals2, atol=1e-14, rtol=0)


def test_low_grid_exp():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fn = np.exp(grid.points)
    fna = grid.antiderivative(fn)
    fna += 1 - fna[npoint // 2]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), np.exp(1) - np.exp(-1), atol=1e-14, rtol=0)
    assert_allclose(fna, fn, atol=1e-14, rtol=0)
    assert_allclose(fnd, fn, atol=1e-10, rtol=0)


def test_tf_grid_exp():
    def tf(t, np):
        u = (1 + t) / 2
        return 10 * np.arctanh(u)**2
    grid = TransformedGrid(tf, 201)
    fn = np.exp(-grid.points)
    fna = grid.antiderivative(fn)
    fna += -1 - fna[0]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), 1.0, atol=1e-13, rtol=0)
    assert_allclose(fna, -fn, atol=1e-7, rtol=0)
    assert_allclose(fnd, -fn, atol=1e-7, rtol=0)


def test_tf_grid_exp_vectorized():
    def tf(t, np):
        u = (1 + t) / 2
        return 15 * np.arctanh(u)**2
    grid = TransformedGrid(tf, 201)
    exponents = np.array([1.0, 0.5, 2.0])
    fns = np.exp(-np.outer(exponents, grid.points))
    fnsa = grid.antiderivative(fns)
    fnsa += (-1 / exponents - fnsa[:, 0])[:, np.newaxis]
    fnsd = grid.derivative(fns)
    assert_allclose(grid.integrate(fns), 1 / exponents, atol=1e-13, rtol=0)
    assert_allclose(fnsa, -fns / exponents[:, np.newaxis], atol=1e-7, rtol=0)
    assert_allclose(fnsd, -fns * exponents[:, np.newaxis], atol=1e-7, rtol=0)
    fns_other = np.exp(-np.outer([1.1, 1.2, 0.8], grid.points))
    integrals2 = np.array([
        [grid.integrate(fn1 * fn2) for fn2 in fns_other]
        for fn1 in fns
    ])
    assert_allclose(grid.integrate(fns, fns_other), integrals2, atol=1e-14, rtol=0)


@pytest.mark.parametrize("nl", sum([[(n, l) for l in range(n)] for n in range(5)], []))
def test_tf_grid_hydrogen_norm(nl):
    """Test the radial grid with the normalization of hydrogen orbitals."""
    def tf(t, np):
        u = (1 + t) / 2
        left = 1e-2
        right = 1e3
        alpha = np.log(right / left)
        return left * (np.exp(alpha * u) - 1)
    grid = TransformedGrid(tf, 101)

    fac = np.math.factorial
    norms = []
    vol = 4 * np.pi * grid.points**2
    n, l = nl
    # This is the same as on Wikipedia, except that the spherical
    # harmonic is replaced by 1/(4*pi).
    normalization = np.sqrt(
        (2 / n)**3 * fac(n - l - 1) / (2 * n * fac(n + l) * 4 * np.pi))
    rho = grid.points * 2 / n
    poly = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    psi = normalization * np.exp(-rho / 2) * rho**l * poly
    norms.append(grid.integrate(psi * psi * vol))
    assert_allclose(norms, 1.0, atol=1e-14, rtol=0)


def test_tf_grid_hydrogen_few():
    def tf(t, np):
        u = (1 + t) / 2
        left = 1e-2
        right = 1e3
        alpha = np.log(right / left)
        return left * (np.exp(alpha * u) - 1)
    grid = TransformedGrid(tf, 101)

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

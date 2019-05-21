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
from scipy.special import eval_genlaguerre

from tinygrid import *


def test_low_grid():
    for npoint in range(10, 20):
        grid = LegendreGrid(npoint)
        assert grid.basis.shape == (npoint, npoint)
        assert grid.basis_inv.shape == (npoint, npoint)
        fn1 = np.random.uniform(0, 1, npoint)
        coeffs = np.dot(grid.basis_inv, fn1)
        fn2 = np.dot(grid.basis, coeffs)
        assert_allclose(fn1, fn2, atol=1e-14)
        fn3 = legval(grid.points, coeffs)
        assert_allclose(fn1, fn3, atol=1e-14)


def test_low_grid_sin():
    npoint = 100
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fn = np.cos(grid.points)
    fni = grid.antiderivative(fn)
    fni -= fni[npoint // 2]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), np.sin(1) - np.sin(-1), atol=1e-14, rtol=0)
    assert_allclose(fni, np.sin(grid.points), atol=1e-14, rtol=0)
    assert_allclose(fnd, -np.sin(grid.points), atol=1e-11, rtol=0)


def test_low_grid_exp():
    npoint = 101
    grid = LegendreGrid(npoint)
    assert_allclose(grid.points[npoint // 2], 0.0, atol=1e-10)
    fn = np.exp(grid.points)
    fni = grid.antiderivative(fn)
    fni += 1 - fni[npoint // 2]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), np.exp(1) - np.exp(-1), atol=1e-14, rtol=0)
    assert_allclose(fni, fn, atol=1e-14, rtol=0)
    assert_allclose(fnd, fn, atol=1e-10, rtol=0)


def test_tf_grid_exp():
    def tf(t, np):
        u = (1 + t) / 2
        return 10 * np.arctanh(u)**2
    grid = TransformedGrid(tf, 201)
    fn = np.exp(-grid.points)
    fni = grid.antiderivative(fn)
    fni += -1 - fni[0]
    fnd = grid.derivative(fn)
    assert_allclose(grid.integrate(fn), 1.0, atol=1e-13, rtol=0)
    assert_allclose(fni, -fn, atol=1e-7, rtol=0)
    assert_allclose(fnd, -fn, atol=1e-7, rtol=0)


def test_tf_grid_hydrogen_norm():
    """Test the radial grid with the normalization of hydrogen orbitals."""
    def tf(t, np):
        u = (1 + t) / 2
        return 1e-4 * np.exp(15 * u)
    grid = TransformedGrid(tf, 201)

    fac = np.math.factorial
    norms = []
    vol = 4 * np.pi * grid.points**2
    for n in range(1, 5):
        for l in range(n):
            # This is the same as on Wikipedia, except that the spherical
            # harmonic is replaced by 1/(4*pi).
            normalization = np.sqrt(
                (2 / n)**3 * fac(n - l - 1) / (2 * n * fac(n + l) * 4 * np.pi))
            rho = grid.points * 2 / n
            poly = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
            psi = normalization * np.exp(-rho / 2) * rho**l * poly
            norms.append(grid.integrate(psi * psi * vol))
    assert_allclose(norms, 1.0, atol=1e-11, rtol=0)


def test_tf_grid_hydrogen_few():
    def tf(t, np):
        u = (1 + t) / 2
        return 1e-4 * np.exp(15 * u)
    grid = TransformedGrid(tf, 201)

    # Solutions of the radial equation (U=R/r)
    psi_1s = np.sqrt(4 * np.pi) * grid.points * np.exp(-grid.points) / np.sqrt(np.pi)
    psi_2s = np.sqrt(4 * np.pi) * grid.points * np.exp(-grid.points / 2) / \
        np.sqrt(2 * np.pi) / 4 * (2 - grid.points)

    # Check norms and energies
    for eps, psi in [(-0.5, psi_1s), (-0.125, psi_2s)]:
        norm = grid.integrate(psi**2)
        assert_allclose(norm, 1.0)
        ekin = grid.integrate(-psi * grid.derivative(psi, 2) / 2)
        assert_allclose(ekin, -eps, atol=1e-6)
        epot = grid.integrate(-psi**2 / grid.points)
        assert_allclose(epot, 2 * eps)

    dot = grid.integrate(psi_1s * psi_2s)
    assert_allclose(dot, 0.0, atol=1e-12, rtol=0)

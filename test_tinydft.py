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

from tinydft import *


def test_char2l():
    assert char2l('s') == 0
    assert char2l('S') == 0
    assert char2l('p') == 1
    assert char2l('P') == 1
    assert char2l('d') == 2
    assert char2l('D') == 2


def test_hydrogen():
    grid = setup_grid(401)
    obasis = setup_obasis(grid, 20)
    overlap = compute_overlap_operator(grid, obasis)
    kin = compute_radial_kinetic_operator(grid, obasis)
    vext = -1 / grid.points
    nai = compute_potential_operator(grid, obasis, vext)
    core = kin + nai
    evals, evecs = eigh(core, overlap)

    ekin = np.einsum('i,ij,j', evecs[:, 0], kin, evecs[:, 0])
    enai = np.einsum('i,ij,j', evecs[:, 0], nai, evecs[:, 0])
    assert_allclose(ekin, 0.5, atol=1e-7)
    assert_allclose(enai, -1.0, atol=1e-7)
    assert_allclose(evals[0], -0.5, atol=1e-7)
    assert_allclose(evals[1], -0.125, atol=1e-7)


def test_tf_grid_poisson():
    # Numerically solve the electrostatic potential of an S-type Slater Density
    grid = setup_grid(401)
    alpha = 0.7
    aux = np.exp(-alpha * grid.points)
    rho = aux * alpha**3 / (8 * np.pi)
    assert_allclose(grid.integrate(4 * np.pi * grid.points**2 * rho), 1.0, atol=1e-11, rtol=0)
    vnum = solve_poisson(grid, rho)
    vann = (1 - aux) / grid.points - alpha * aux / 2
    assert_allclose(vnum[-1], 1.0 / grid.points[-1], atol=1e-11, rtol=0)
    assert_allclose(vnum, vann, atol=1e-7, rtol=0)


def test_interpret_econf():
    occups = interpret_econf('1s1 2s2 2p3.0')
    assert len(occups) == 2
    assert len(occups[0]) == 2
    assert len(occups[1]) == 1
    assert occups[0][0] == 1.0
    assert occups[0][1] == 2.0
    assert occups[1][0] == 3.0

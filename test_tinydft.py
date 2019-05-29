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

import numpy as np
from numpy.testing import assert_allclose
import pytest

from tinydft import scf_atom, setup_grid, solve_poisson, char2angqn, klechkowski, interpret_econf


def test_char2angqn():
    assert char2angqn('s') == 0
    assert char2angqn('S') == 0
    assert char2angqn('p') == 1
    assert char2angqn('P') == 1
    assert char2angqn('d') == 2
    assert char2angqn('D') == 2


@pytest.mark.parametrize("atnum", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
def test_poisson(atnum):
    # Numerically solve the electrostatic potential of an S-type Slater Density
    grid = setup_grid()
    alpha = 2 * atnum
    aux = np.exp(-alpha * grid.points)
    rho = aux * alpha**3 / (8 * np.pi)
    assert_allclose(grid.integrate(4 * np.pi * grid.points**2 * rho), 1.0, atol=1e-11, rtol=0)
    vnum = solve_poisson(grid, rho)
    vann = (1 - aux) / grid.points - alpha * aux / 2
    assert_allclose(vnum[-1], 1.0 / grid.points[-1], atol=1e-11, rtol=0)
    assert_allclose(vnum, vann, atol=2e-5, rtol=0)


def test_interpret_econf1():
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


def test_interpret_econf2():
    occups = interpret_econf('1s1 3s2 3p2.5')
    assert len(occups) == 2
    assert len(occups[0]) == 3
    assert len(occups[1]) == 2
    assert occups[0][0] == 1.0
    assert occups[0][1] == 0.0
    assert occups[0][2] == 2.0
    assert occups[1][0] == 0.0
    assert occups[1][1] == 2.5
    with pytest.raises(TypeError):
        interpret_econf('1s1 2s0')
    with pytest.raises(TypeError):
        interpret_econf('1s1 2s-2')


def test_klechkowski():
    assert klechkowski(1) == '1s1'
    assert klechkowski(1.5) == '1s1.5'
    assert klechkowski(2) == '1s2'
    assert klechkowski(10) == '1s2 2s2 2p6'
    assert klechkowski(23) == '1s2 2s2 2p6 3s2 3p6 4s2 3d3'
    assert klechkowski(118) == ('1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 '
                                '6s2 4f14 5d10 6p6 7s2 5f14 6d10 7p6')


@pytest.mark.parametrize("atnum", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121])
def test_atom(atnum, num_regression, grid_basis):
    econf = klechkowski(atnum)
    occups = interpret_econf(econf)
    grid, basis = grid_basis
    energies, rho = scf_atom(atnum, occups, grid, basis, nscf=100)
    nelec = grid.integrate(4 * np.pi * grid.points**2 * rho)
    assert_allclose(nelec, atnum, atol=1e-8, rtol=0)
    num_regression.check(
        {'energies': energies},
        default_tolerance={'rtol': 1e-9, 'atol': 0},
    )

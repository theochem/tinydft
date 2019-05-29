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
from scipy.linalg import eigh
import pytest

from tinydft import setup_grid
from tinybasis import Basis
from test_tinygrid import get_hydrogenic_solutions


@pytest.mark.parametrize("atnum", [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
@pytest.mark.parametrize("angqn", [0, 1, 2, 3, 4, 5, 6])
def test_hydrogenic_op(atnum, angqn, grid_basis):
    grid, basis = grid_basis
    # Created modified copies of the operators. Do not modificy in place to
    # avoid side-effects.
    kin = basis.kin_rad.copy()
    if angqn > 0:
        angmom_factor = (angqn * (angqn + 1)) / 2
        kin += basis.kin_ang * angmom_factor
    ext = basis.ext * atnum
    evals, evecs = eigh(kin + ext, basis.olp)
    psis = get_hydrogenic_solutions(grid, atnum, angqn)
    for i, (priqn, factor, psi) in enumerate(psis):
        case = "i={} priqn={}".format(i, priqn)
        # Same test for the numerical solution
        dot = abs(grid.integrate(psi, np.dot(evecs[:, i], basis.fnvals)))
        norm = np.einsum('i,ij,j', evecs[:, i], basis.olp, evecs[:, i])
        ekin = np.einsum('i,ij,j', evecs[:, i], kin, evecs[:, i])
        eext = np.einsum('i,ij,j', evecs[:, i], ext, evecs[:, i])
        assert_allclose(dot, 1, atol=0, rtol=1e-7, err_msg=case)
        assert_allclose(norm, 1, atol=0, rtol=1e-8, err_msg=case)
        assert_allclose(eext, -factor, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(ekin, factor / 2, atol=0, rtol=1e-5, err_msg=case)
        assert_allclose(evals[i], -factor / 2, atol=0, rtol=3e-6, err_msg=case)


def test_integral_regression(num_regression):
    grid = setup_grid()
    basis = Basis(grid, 1e-1, 1e2, 5)
    assert basis.olp.shape == (5, 5)
    assert basis.kin_rad.shape == (5, 5)
    assert basis.kin_ang.shape == (5, 5)
    assert basis.ext.shape == (5, 5)
    assert_allclose(np.diag(basis.olp) - 1, 0.0, atol=1e-15, rtol=0)
    num_regression.check(
        {'olp': basis.olp.ravel(),
         'kin_rad': basis.kin_rad.ravel(),
         'kin_ang': basis.kin_ang.ravel(),
         'ext': basis.ext.ravel()},
        default_tolerance={'rtol': 1e-15, 'atol': 0},
    )

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
"""Configuration of pytest."""


import pytest

from tinydft import setup_grid
from tinybasis import Basis


@pytest.fixture(scope="session")
def grid_basis():
    """Set up default integration grid and basis set.

    This fixture makes it possible to initiliase and precompute these objects
    only once for all tests.
    """
    grid = setup_grid()
    basis = Basis(grid)
    return grid, basis

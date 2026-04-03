# SPDX-FileCopyrightText: © 2023 Tiny DFT Development Team <https://github.com/molmod/acid/blob/main/AUTHORS.md>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration of pytest."""

import pytest

from tinydft.basis import Basis
from tinydft.grid import setup_grid


@pytest.fixture(scope="session")
def grid_basis():
    """Set up default integration grid and basis set.

    This fixture makes it possible to initiliase and precompute these objects
    only once for all tests.
    """
    grid = setup_grid()
    basis = Basis(grid)
    return grid, basis

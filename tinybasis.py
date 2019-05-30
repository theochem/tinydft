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
"""Radial orbital basis of Gaussian primitives and their integrals for TinyDFT."""


from functools import wraps

import numpy as np
from numpy.testing import assert_allclose

from tinygrid import TransformedGrid


def memoize(method):
    """Wrap a function such that its result is computed only once.

    See https://en.wikipedia.org/wiki/Memoization
    """
    @wraps(method)
    def wrapper(obj) -> np.ndarray:
        attrname = '_' + method.__name__
        result = getattr(obj, attrname, None)
        if result is None:
            result = method(obj)
        setattr(obj, attrname, result)
        return result
    return wrapper


class Basis:
    """Gaussian radial orbital basis set, precomputed integrals and integral evaluation.

    Results of integrals that take no arguments (overlap, kin_rad, kin_ang and
    ext) are memoized. After construction, it is assumed that the basis
    functions are not modified.

    Attributes
    ----------
    grid
        The radial numerical integration grid.
    alphas
        Gaussian exponents.
    fnvals
        Normalized basis functions evaluated on the grid.
    normalizations
        Vector with normalization constants for the Gaussians.
    olp
        Overlap matrix
    kin_rad
        Radial kinetic energy operator.
    kin_ang
        Angular kinetic energy operator for angqn=1.
    ext
        Operator for the interaction with a proton, i.e. the external field.

    """

    def __init__(self, grid: TransformedGrid, alphamin: float = 1e-6,
                 alphamax: float = 1e8, nbasis: int = 80):
        """Initialize a basis.

        Parameters
        ----------
        grid
            The radial integration grid.
        alphamin
            The lowest Gaussian exponent.
        alphamax
            The highest Gaussian exponent.
        nbasis
            The number of basis functions.

        """
        self.grid = grid
        self.alphas = 10**np.linspace(np.log10(alphamin), np.log10(alphamax), nbasis)
        self.fnvals = np.exp(-np.outer(self.alphas, grid.points**2)) * grid.points
        self.normalizations = np.sqrt(np.sqrt(self.alphas))**3 * np.sqrt(np.sqrt(2 / np.pi) * 8)
        self.fnvals *= self.normalizations[:, np.newaxis]
        assert_allclose(np.sqrt(grid.integrate(self.fnvals**2)), 1.0, atol=7e-14, rtol=0)

    @property
    def nbasis(self) -> int:
        """Return the number of basis functions."""
        return self.fnvals.shape[0]

    @property  # type: ignore
    @memoize
    def olp(self) -> np.ndarray:
        """Return the overlap matrix."""
        alpha_sums = np.add.outer(self.alphas, self.alphas)
        alpha_prods = np.outer(self.alphas, self.alphas)
        return (2 * np.sqrt(2)) * (alpha_prods)**0.75 / alpha_sums**1.5

    @property  # type: ignore
    @memoize
    def kin_rad(self) -> np.ndarray:
        """Return the radial kinetic energy operator."""
        alpha_sums = np.add.outer(self.alphas, self.alphas)
        alpha_prods = np.outer(self.alphas, self.alphas)
        return np.sqrt(72) * alpha_prods**1.75 / alpha_sums**2.5

    @property  # type: ignore
    @memoize
    def kin_ang(self) -> np.ndarray:
        """Return the angular kinetic energy operator for angqn=1."""
        alpha_sums = np.add.outer(self.alphas, self.alphas)
        alpha_prods = np.outer(self.alphas, self.alphas)
        return np.sqrt(32) * (alpha_prods)**0.75 / np.sqrt(alpha_sums)

    @property  # type: ignore
    @memoize
    def ext(self) -> np.ndarray:
        """Return the operator for the interaction with the external field, i.e. a proton."""
        alpha_sums = np.add.outer(self.alphas, self.alphas)
        alpha_prods = np.outer(self.alphas, self.alphas)
        return -np.sqrt(32 / np.pi) / alpha_sums * alpha_prods**0.75

    def pot(self, pot: np.ndarray) -> np.ndarray:
        """Return the operator for the interaction with a potential on a grid."""
        return self.grid.integrate(self.fnvals, self.fnvals, pot)

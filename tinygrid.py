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
"""Tiny numerical integration library for Tiny DFT."""

from typing import List, Callable

import numpy as np
from numpy.polynomial.legendre import legvander, legder, legint, leggauss


__all__ = ['LegendreGrid', 'TransformedGrid']


class BaseGrid:
    """Basic grid API, only for (vectorized) definite integrals."""

    def __init__(self, points: np.ndarray, weights: np.ndarray):
        self.points = points
        self.weights = weights

    def integrate(self, *multi_fnvals: np.ndarray) -> np.ndarray:
        """Compute the definite integrals (of products) of functions.

        Parameters
        ----------
        fnvals1, fnvals2, ...
            Each argument is an N-dimensional array of function values, where
            the last index corresponds to the grid points. The arrays are
            contracted over their last index and the resulting functions are
            all integrated. This allows many products of functions to be
            integrated in one shot.

        Returns
        -------
        integrals
            All integrals of products of functions. The shape is
            ``sum(fnvals.shape[:-1] for fnvals in multi_fnvals)``.

        """
        args = [self.weights, [0]]
        sublistout: List[int] = []
        counter = 1
        for fnvals in multi_fnvals:
            args.append(fnvals)
            nkeep = fnvals.ndim - 1
            sublist = list(range(counter, counter + nkeep)) + [0]
            sublistout += sublist[:-1]
            args.append(sublist)
            counter += nkeep
        args.append(sublistout)
        return np.einsum(*args, optimize=True)


class LegendreGrid(BaseGrid):
    """Standard Gauss-Legendre grid.

    The derivative and antiderivative are implemented with the spectral method.

    All methods in this class are designed to vectorize operations. Each method
    argument can be an N-dimensional array, of which the last index corresponds
    to grid points or Legendre polynomials. The functions vectorize over the
    preceding indices.
    """

    def __init__(self, npoint: int):
        """Initialize a Gauss-Legendre grid with given number of points."""
        BaseGrid.__init__(self, *leggauss(npoint))
        # Basis functions are used to transform from grid data to Legendre
        # coefficients and back.
        self.basis = legvander(self.points, npoint - 1)
        # pylint: disable=invalid-name
        U, S, Vt = np.linalg.svd(self.basis)
        self.basis_inv = np.einsum('ji,j,kj->ik', Vt, 1 / S, U)

    def tocoeffs(self, fnvals: np.ndarray) -> np.ndarray:
        """Convert function values on a grid to Legendre coefficients."""
        return np.dot(fnvals, self.basis_inv.T)

    def tofnvals(self, coeffs: np.ndarray) -> np.ndarray:
        """Convert Legendre coefficients to function values on a grid."""
        return np.dot(coeffs, self.basis.T[:coeffs.shape[-1]])

    def antiderivative(self, fnvals: np.ndarray) -> np.ndarray:
        """Return the antiderivative."""
        coeffs = self.tocoeffs(fnvals)
        coeffs_int = legint(coeffs, axis=-1)
        return self.tofnvals(coeffs_int[..., :-1])

    def derivative(self, fnvals: np.ndarray) -> np.ndarray:
        """Return the derivative."""
        coeffs = self.tocoeffs(fnvals)
        coeffs_der = legder(coeffs, axis=-1)
        return self.tofnvals(coeffs_der)


class TransformedGrid(BaseGrid):
    """A custom transformation of the Legendre grid.

    For vectorization, see documentation string of the Legendre class.
    """

    def __init__(self, transform: Callable, npoint: int):
        """Initialize the transformed grid.

        Parameters
        ----------
        transform
            A function taking two arguments: (i) array of Legendre points and
            (ii) the numpy wrapper to use, which enables algorithmic
            differentiation. The result is the array with transformed grid
            points.
        npoint
            The number of grid points.

        """
        from autograd import elementwise_grad
        import autograd.numpy as agnp
        self.legendre_grid = LegendreGrid(npoint)
        points = transform(self.legendre_grid.points, np)
        # Compute the Jacobian of the grid transformation and update weights.
        # pylint: disable=no-value-for-parameter
        self.derivs = abs(elementwise_grad(transform)(self.legendre_grid.points, agnp))
        weights = self.legendre_grid.weights * self.derivs
        BaseGrid.__init__(self, points, weights)

    def antiderivative(self, fnvals: np.ndarray) -> np.ndarray:
        """Return the antiderivative."""
        return self.legendre_grid.antiderivative(fnvals * self.derivs)

    def derivative(self, fnvals: np.ndarray) -> np.ndarray:
        """Return the derivative."""
        return self.legendre_grid.derivative(fnvals) / self.derivs

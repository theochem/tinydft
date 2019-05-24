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

from __future__ import print_function, division  # Py 2.7 compatibility

import numpy as np
from numpy.polynomial.legendre import legvander, legder, legint, leggauss


__all__ = ['LegendreGrid', 'TransformedGrid']


class LegendreGrid:
    """Standard Gauss-Legendre grid.

    The derivative and antiderivative are implemented with the spectral method.
    """

    def __init__(self, npoint):
        """Initialize a Gauss-Legendre grid with given number of points."""
        self.points, self.weights = leggauss(npoint)

        # Basis functions are used to obtain the Legendre coefficients
        # given a function on a grid. (Mind the normalization.)
        self.basis = legvander(self.points, npoint - 1)
        U, S, Vt = np.linalg.svd(self.basis)
        self.basis_inv = np.einsum('ji,j,kj->ik', Vt, 1 / S, U)

    def integrate(self, fn):
        """Compute the definite integral of fn."""
        return np.dot(self.weights, fn)

    def antiderivative(self, fn, order=1):
        """Return the antiderivative to given order."""
        coeffs = np.dot(self.basis_inv, fn)
        coeffs_int = legint(coeffs, order)
        return np.dot(self.basis, coeffs_int[:-1])

    def derivative(self, fn, order=1):
        """Return the derivative to given order."""
        coeffs = np.dot(self.basis_inv, fn)
        coeffs_der = legder(coeffs, order)
        return np.dot(self.basis[:, :-1], coeffs_der)


class TransformedGrid:
    """A custom transformation of the Legendre grid."""

    def __init__(self, transform, npoint):
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
        import autograd.numpy as np
        self.legendre_grid = LegendreGrid(npoint)
        self.points = transform(self.legendre_grid.points, np)
        self.derivs = abs(elementwise_grad(transform)(self.legendre_grid.points, np))
        self.weights = self.legendre_grid.weights * self.derivs

    def integrate(self, fn):
        """Compute the definite integral of fn."""
        return np.dot(self.weights, fn)

    def antiderivative(self, fn, order=1):
        """Return the antiderivative to given order."""
        for _ in range(order):
            fn = self.legendre_grid.antiderivative(fn * self.derivs)
        return fn

    def derivative(self, fn, order=1):
        """Return the derivative to given order."""
        for _ in range(order):
            fn = self.legendre_grid.derivative(fn) / self.derivs
        return fn

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

import numpy as np
from numpy.polynomial.chebyshev import chebvander, chebder, chebint, chebval


__all__ = ['ChebyGrid', 'TransformedGrid']


class ChebyGrid:
    """Standard Gauss-Chebyshev grid.

    The derivative and antiderivative are implemented with the spectral method.
    """

    def __init__(self, npoint):
        """Initialize a Chebyshev-Gauss grid with given number of points."""
        self.points = -np.cos(np.pi * (np.arange(npoint) + 0.5) / npoint)
        self.weights = np.pi / npoint * np.sqrt(1 - self.points**2)

        # Basis functions are used to obtain the Chebyshev coefficients
        # given a function on a grid. (Mind the normalization.)
        # This could be made more efficient with an FFT approach.
        self.basis = chebvander(self.points, npoint - 1).T
        self.basis[0] /= npoint
        self.basis[1:] /= npoint / 2

    def integrate(self, fn):
        """Compute the definite integral of fn."""
        return np.dot(self.weights, fn)

    def antiderivative(self, fn, order=1):
        """Return the antiderivative to given order."""
        coeffs = np.dot(self.basis, fn)
        coeffs_int = chebint(coeffs, order)
        return chebval(self.points, coeffs_int)

    def derivative(self, fn, order=1):
        """Return the derivative to given order."""
        coeffs = np.dot(self.basis, fn)
        coeffs_der = chebder(coeffs, order)
        return chebval(self.points, coeffs_der)


class TransformedGrid:
    """A custom transformation of the Chebyshev grid."""

    def __init__(self, transform, npoint):
        """Initialize the transformed grid.

        Parameters
        ----------
        transform
            A function taking two arguments: (i) array of Chebyshev points and
            (ii) the numpy wrapper to use, which enables algorithmic
            differentiation. The result is the array with transformed grid
            points.
        npoint
            The number of grid points.

        """
        from autograd import elementwise_grad
        import autograd.numpy as np
        self.cheby_grid = ChebyGrid(npoint)
        self.points = transform(self.cheby_grid.points, np)
        self.derivs = abs(elementwise_grad(transform)(self.cheby_grid.points, np))
        self.weights = self.cheby_grid.weights * self.derivs

    def integrate(self, fn):
        """Compute the definite integral of fn."""
        return np.dot(self.weights, fn)

    def antiderivative(self, fn, order=1):
        """Return the antiderivative to given order."""
        for i in range(order):
            fn = self.cheby_grid.antiderivative(fn * self.derivs)
        return fn

    def derivative(self, fn, order=1):
        """Return the derivative to given order."""
        for i in range(order):
            fn = self.cheby_grid.derivative(fn) / self.derivs
        return fn

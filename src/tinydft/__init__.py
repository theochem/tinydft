# Tiny DFT is a minimalistic atomic DFT implementation.
# Copyright (C) 2024 The Tiny DFT Development Team
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
"""Tiny DFT is a minimalistic atomic DFT implementation.

It only supports closed-shell calculations with pure local (not even semi-local)
functionals. One has to fix the occupations numbers of the atomic orbitals a
priori.

Atomic units are used throughout.
"""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0a-dev"
    __version_tuple__ = (0, 0, 0, "a-dev")

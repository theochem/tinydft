# SPDX-FileCopyrightText: © 2023 Tiny DFT Development Team <https://github.com/molmod/acid/blob/main/AUTHORS.md>
# SPDX-License-Identifier: GPL-3.0-or-later
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

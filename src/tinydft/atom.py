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
"""Utilities for setting up atomic calculations.

This is used by the demos and the unit tests.
"""

import numpy as np

__all__ = ("char2angqn", "interpret_econf", "klechkowski")


ANGMOM_CHARACTERS = "spdfghiklmnoqrtuvwxyzabce"


def char2angqn(char: str) -> int:
    """Return the angular momentum quantum number corresponding to a character."""
    return ANGMOM_CHARACTERS.index(char.lower())


def interpret_econf(econf: str) -> list[np.ndarray]:
    """Translate econf strings into occupation numbers per angular momentum.

    Parameters
    ----------
    econf
        An electronic configuration string, e.g '1s2 2s2 2p3.5' for oxygen with
        7.5 electons.

    Returns
    -------
    occups
        A list of arrays, one array per angular momentum (in the order s, p, d,
        ...). Each array contains the electron occupation numbers for the radial
        orbitals for a given angular momentum. The maximum occupation number is
        ``2 * (2 * angqn + 1)``. The radial orbitals for a given angular
        momentum quantum number are assumed to be ordered from low to high. The
        occupation number arrays are truncated after the highest occupied
        orbital, i.e. remaining zero occupation numbers are not included.

    """
    occups: list[list[float]] = []
    for key in econf.split():
        occup = float(key[2:])
        if occup <= 0:
            raise ValueError(
                "Occuptions in the electronic configuration must be strictly positive."
            )
        priqn = int(key[0])
        angqn = char2angqn(key[1])
        if occup > 2 * (2 * angqn + 1):
            raise ValueError(
                "Occuptions in the electronic configuration must not exceed 2 * (2 * angqn + 1)."
            )
        # Add more angular momenta if needed.
        while len(occups) < angqn + 1:
            occups.append([])
        # Add more energy levels
        i = priqn - angqn - 1
        while len(occups[angqn]) < i + 1:
            occups[angqn].append(0.0)
        occups[angqn][i] = occup
    return [np.array(angqn_occups) for angqn_occups in occups]


def format_econf(econf: str) -> str:
    """Return an electron configuration suitable for printing."""
    for noble_atnum in NOBLE_ATNUMS[::-1]:
        econf_noble = klechkowski(noble_atnum)
        if econf_noble in econf and econf != econf_noble:
            return econf.replace(econf_noble, f"[{SYMBOLS[noble_atnum]}]")
    return econf


def klechkowski(nelec: float) -> str:
    """Return the atomic electron configuration by following the Klechkowski rule."""
    priqn_plus_angqn = 1
    words = []
    while nelec > 0:
        # start with highest possible angqn for given priqn+angqn
        angqn = (priqn_plus_angqn - 1) // 2
        priqn = priqn_plus_angqn - angqn
        while nelec > 0 and angqn >= 0:
            nelec_orb = min(nelec, 2 * (2 * angqn + 1))
            nelec -= nelec_orb
            words.append(f"{priqn}{ANGMOM_CHARACTERS[angqn]}{nelec_orb}")
            angqn -= 1
            priqn += 1
        priqn_plus_angqn += 1
    return " ".join(words)


SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


NOBLE_ATNUMS = [2, 10, 18, 36, 54, 86]

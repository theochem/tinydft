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
"""Plot the energy as function of number of electrons for Carbon: N = 0 to 6."""

import matplotlib.pyplot as plt

from tinybasis import Basis
from tinydft import scf_atom
from tinygrid import setup_grid
from program_mendelejev import klechkowski, interpret_econf


def main():
    """Make a plot of the energy of Carbon as function of the number of electrons."""
    # Compute the energies.
    atnum = 6
    grid = setup_grid()
    basis = Basis(grid)
    energies_scan = [0.0]
    for nelec in range(1, 7):
        econf = klechkowski(nelec)
        occups = interpret_econf(econf)
        energies, _rhos = scf_atom(atnum, occups, grid, basis)
        energies_scan.append(energies[0])

    # Make a nice plot.
    plt.rcParams['font.sans-serif'] = ["Fira Sans"]
    fig, ax = plt.subplots(figsize=(4, 2.25))
    ax.plot(energies_scan, "o")
    ax.set_xlabel("Number of electrons")
    ax.set_ylabel(r"Electronic energy [$\mathrm{E_h}$]")
    ax.set_title("Carbon (HFS functional, TinyDFT)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("energy_verus_nelec_carbon.png", dpi=300)


if __name__ == '__main__':
    main()

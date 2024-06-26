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
"""Plot the energy as function of number of electrons for Carbon: N = 0 to 6."""

import matplotlib.pyplot as plt

from .atom import interpret_econf, klechkowski
from .basis import Basis
from .dft import scf_atom
from .grid import setup_grid


def main(fn_out: str = "energy_verus_nelec_carbon.png"):
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
    fig, ax = plt.subplots(figsize=(4, 2.25))
    ax.plot(energies_scan, "o")
    ax.set_xlabel("Number of electrons")
    ax.set_ylabel(r"Electronic energy [$\mathrm{E_h}$]")
    ax.set_title("Carbon (HFS functional, TinyDFT)")
    fig.savefig(fn_out)
    plt.close(fig)


if __name__ == "__main__":
    main()

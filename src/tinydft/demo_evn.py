# SPDX-FileCopyrightText: © 2023 Tiny DFT Development Team <https://github.com/molmod/acid/blob/main/AUTHORS.md>
# SPDX-License-Identifier: GPL-3.0-or-later
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

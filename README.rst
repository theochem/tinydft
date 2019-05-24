Tiny DFT
########

Tiny DFT is a minimalistic atomic Density Functional Theory (DFT) code, mainly
for educational purposes. It only supports spherical closed-shell atoms (with
fractional occupations to obtain a spherical density) and local
exchange-correlation functionals (at the moment only Dirac exchange).

The code is designed with the following criteria in mind:

- It depends only on established scientific Python libraries: numpy_, scipy_,
  matplotlib_ and (the lesser known) autograd_. The latter is a library for
  algorithmic differentiation, used to computed the analytic
  exchange(-correlation) potential and the grid transformation.

- The numerical integration and differentiation algorithms should be accurate
  enough to at least 3 significant digits in the total energy, but in many cases
  the numerical accuracy is better. (The pseudo-spectral method with Legendre
  polynomials is used.)

- The total number of lines should be minimal and the source-code should be easy
  to understand, provided some background in DFT and spectral methods.


"Installation"
==============

1) Make sure you have the dependencies installed: Python 3 and fairly recent
   versions of numpy_ (>= 1.4.0), scipy_ (>=1.0.0), matplotlib_ (>= 2.2.4) and
   autograd_ (>=1.2). In case of doubt, ask some help from your local Python
   guru. If you have Python 3, you can always install or upgrade the other
   dependencies in your user account with pip:

   .. code-block:: bash

        python3 -m pip install numpy scipy autograd --upgrade

   Packages from your Linux distribution or the Conda package manager should
   also work.

2) Download Tiny DFT. This can be done with your browser, after which you unpack
   the archive: https://github.com/theochem/tinydft/archive/master.zip.
   Or you can use git:

   .. code-block:: bash

        git clone https://github.com/theochem/tinydft.git
        cd tinydft

Usage
=====

To run an atomic DFT calculation, just execute the tinydft.py script:

.. code-block::

    python3 tinydft.py

This generates some screen output with energy contributions and a figure
``rho.png`` with the radial electron density on a semi-log grid. To modify the
settings for this calculation, you have to directly edit the source code.

When you make serious modifications to Tiny DFT, you can run the unit tests to
make sure the original features still work. For this, you first need to install
pytest_.

.. code-block:: bash

    # Install pytest in case you don't have it yet.
    python3 -m pip install pytest --upgrade
    pytest


Programming assignments
=======================

In order of increasing difficulty:

1) Change ``tinydft.py`` to also make plots of the radial probability density,
   the occupied orbitals, the potentials (external, Kohn-Sham) with horizontal
   lines for the (higher but still negative) energy levels. Does this code
   reproduce the Rydberg spectral series well? (See
   https://en.wikipedia.org/wiki/Hydrogen_spectral_series#Rydberg_formula)

2) Write a driver script ``driver.py``, which uses ``tinydft.py`` as a python
   module to compute the ionization potentials and electron affinities of all
   atoms in the periodic table. (See how far you can get before the numerical
   algorithms break.) Implement *Madelung energy ordering rule* to set the
   electronic configuration. See
   https://en.wikipedia.org/wiki/Aufbau_principle#Madelung_energy_ordering_rule

3) Add a second unit test for the Poisson solver in ``test_tinydft.py``. The
   current unit test checks if the Poisson solver can correctly compute the
   electrostatic potential of a spherical exponential density distribution
   (Slater-type). Add a similar test for a Gaussian density distribution. Use
   the ``erf`` function from ``scipy.special`` to compute the analytic result.
   See
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html#scipy.special.erf

4) Change the external potential to take into account the finite extent of the
   nucleus. You can use a Gaussian density distribution. Write a
   python script that combines isotope abundancies from NUBASE2016
   (http://amdc.in2p3.fr/nubase/nubase2016.txt) and nucleotide radii from ADNDT
   (https://www-nds.iaea.org/radii/) to build a table of abundance-averaged
   nuclear radii.

5) Add a correlation energy density the function ``excfunction`` and check if it
   improve the results in assignment (2). The following correlation functional
   has a good compromise between simplicity and accuracy:
   https://aip.scitation.org/doi/10.1063/1.4958669 and
   https://aip.scitation.org/doi/full/10.1063/1.4964758

6) Replace the current ``econf`` argument of the ``main`` function by a number
   of electrons. Use the Aufbau principle to assign occupation numbers at each
   SCF iteration instead of the currently fixed values for each orbital.
   See https://en.wikipedia.org/wiki/Aufbau_principle Try not to use the
   Klechkowsky (or Madelung) rules, but just use the orbital energies to
   at each SCF iteration to find the lowest-energy orbitals.

7) Implement an SCF convergence test, which checks if the new Fock operator, in
   the basis of occupied orbitals from a previous iteration, is diagonal with
   orbital energies from the previous iteration on the diagonal

8) Implement the zeroth-order regular approximation to the Dirac equation
   (ZORA) to the code. ZORA needs a pro-atomic Kohn-Sham potential as input,
   which remains fixed during the SCF cycle. Add an outer loop where the first
   iteration is without ZORA and subsequent iterations use the Kohn-Sham
   potential from the previous SCF loop as pro-density for ZORA. (This requires
   the changes from assignment 4 to be implemented.)

   In ZORA, the following operator should be added to the Hamiltonian:

   .. image:: zora.png
     :alt: t_{ab} = \int (\nabla \chi_a) (\nabla \chi_b) \frac{v_{KS}(\mathbf{r})}{4/\alpha^2 - 2v_{KS}(\mathbf{r})} \mathrm{d}\mathbf{r}
     :align: center

   where the first factors are the gradients of the basis functions (similar to
   the kinetic energy operator). The Kohn-Sham potential from the previous
   outer iteration can be used. The parameter alpha is the dimensionless inverse
   fine-structure constant, see
   https://physics.nist.gov/cgi-bin/cuu/Value?alphinv and
   https://docs.scipy.org/doc/scipy/reference/constants.html (``inverse
   fine-structure constant``). Before ZORA can be implemented, the formula
   needs to be worked out in spherical coordinates, separating it in a
   radial and an angular contribution.

9) Extend the program to perform unrestricted Spin-polarized DFT calculations.
   (Assignment 5 should done prior to this one.) In addition to the Aufbau rule,
   you now also have to implement the Hund rule. You also need to keep track of
   spin-up and spin-down orbitals. The original code uses the angular momentum
   quantum number as keys in the ``eps_orbs_u`` dictionary. Instead, you can
   now use ``(l, spin)`` keys.

10) Extend the program to support Hartree-Fock exchange.

11) Extend the program to support (meta) generalized gradient functionals.


.. _numpy: https://www.numpy.org/

.. _scipy: https://www.scipy.org/

.. _matplotlib: https://matplotlib.org/

.. _autograd: https://github.com/HIPS/autograd/

.. _pytest: https://pytest.org/

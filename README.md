# GNLStools 

`GNLStools.py` is a Python module containing data structures and functions for
simulation and analysis of the propagation dynamics of ultrashort laserpulses
in nonlinear waveguides, described by the generalized nonlinear Schrödinger
equation. 

The provided software implements the effects of linear dispersion, pulse
self-steepening, and the Raman effect. Input pulse shot noise can be included
using one-photon-per mode quantum noise models, and coherence properties of the
resulting spectra can be calculated.

We include an example, demonstrating the functionality of the software by
reproducing results for a supercontinuum generation process in a photonic
crystal fiber, documented in the scientific literature.

## Prerequisites

`GNLStools` is developed under python3 (version 3.9.7) and requires the
functionality of 

* numpy (1.21.2)
* scipy (1.7.0)

Further, the figure generation scripts included with the examples require the
functionality of

* matplotlib (3.4.3)

`GNLStools` can be used as an extension module for
[py-fmas](https://github.com/omelchert/py-fmas), allowing a user to take
advantage of variable stepsize z-propagation algorithms.

## Availability of the software

The `GNLStools` presented here are derived from our research software and meant
to work as a (system-)local software tool. There is no need to install it once
you got a local
[clone](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
of the repository, e.g. via

``$ git clone https://github.com/omelchert/GNLStools``

## License 

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

This work received funding from the Deutsche Forschungsgemeinschaft  (DFG)
under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD
(Photonics, Optics, and Engineering – Innovation Across Disciplines) (EXC 2122,
projectID 390833453).

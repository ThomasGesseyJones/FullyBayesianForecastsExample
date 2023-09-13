================================
Fully Bayesian Forecasts Example
================================

Overview
--------

:Name: Fully Bayesian Forecasts Example
:Author: Thomas Gessey-Jones
:Version: 0.1.2
:Homepage: https://github.com/ThomasGesseyJones/FullyBayesianForecastsExample
:Paper: TBD

.. image:: https://img.shields.io/badge/python-3.8-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ThomasGesseyJones/ErrorAffirmations/blob/main/LICENSE
   :alt: License information

Example of a Fully Bayesian Forecast using an `Evidence Network <https://ui.adsabs.harvard.edu/abs/2023arXiv230511241J/abstract>`__.
This code also replicates the analysis presented in
`Gessey-Jones et al. (2023) <TBD>`__.
It thus serves the dual purposes of providing an example code base others
can modify for their own Fully Bayesian Forecasts and providing a
reproducible analysis for the letter.

The overall goal of the code is to produce a Fully Bayesian Forecast for
the chance of a `REACH <https://ui.adsabs.harvard.edu/abs/2022NatAs...6..984D/abstract>`__ like experiment
detecting the 21-cm global signal, given a noise level. It also produces
figures showing how this conclusion changes with fixed astrophysical parameters
and provides a way to validate the forecast through blind coverage
tests and comparison to `PolyChord <https://ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H/abstract>`__.



Installation
------------

As an example analysis code the repository is intended to be installed locally
by cloning the repository. To do this run the following command in the terminal

.. code:: bash

    git clone git@github.com:ThomasGesseyJones/FullyBayesianForecastsExample.git

This will create a local copy of the repository. The three main scripts can
then be run from the terminal (see below).


Licence and Citation
--------------------

The software is free to use on the MIT open source license.
If you use the software for academic purposes then we request that you cite
the `letter <TBD>`__ ::

   TBD

If you are using Bibtex you can use the following to cite the letter

.. code:: bibtex

    TBD

Note some of the packages used (see below) in this code have their own licenses that
require citation when used for academic purposes (e.g. `globalemu <https://github.com/htjb/globalemu>`__ and
`pypolychord <https://github.com/PolyChord/PolyChordLite>`__). Please check the licenses of these packages for more details.


Requirements
------------

To run the code you will need to following additional packages:

- `globalemu <https://pypi.org/project/globalemu/>`__
- `tensorflow <https://pypi.org/project/tensorflow/>`__
- `numpy <https://pypi.org/project/numpy/>`__
- `keras <https://pypi.org/project/keras/>`__
- `matplotlib <https://pypi.org/project/matplotlib/>`__
- `nvidia-cudnn-cu11 <https://pypi.org/project/nvidia-cudnn-cu11/>`__
- `pandas <https://pypi.org/project/pandas/>`__
- `PyYAML <https://pypi.org/project/PyYAML/>`__
- `pypolychord <https://github.com/PolyChord/PolyChordLite>`__
- `scipy <https://pypi.org/project/scipy/>`__
- `mpi4py <https://pypi.org/project/mpi4py/>`__

The code was developed using python 3.8. It has not been tested on other versions
of python. Exact versions of the packages used in our analysis
can be found in the
`requirements.txt <https://github.com/ThomasGesseyJones/FullyBayesianForecastsExample/blob/main/requirements.txt>`__ file
for reproducibility.

Additional packages that were used for linting, versioning, and pre-commit hooks
are also listed in the requirements.txt file.

Issues and Questions
--------------------

If you have any issues or questions about the code please raise an
`issue <https://github.com/ThomasGesseyJones/FullyBayesianForecastsExample/issues>`__
on the github page.

Alternatively you can contact the author directly at
`tg400@cam.ac.uk <mailto:tg400@cam.ac.uk>`__.


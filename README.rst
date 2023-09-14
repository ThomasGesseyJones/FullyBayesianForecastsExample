===============================
Fully Bayesian Forecast Example
===============================

Overview
--------

:Name: Fully Bayesian Forecast Example
:Author: Thomas Gessey-Jones
:Version: 0.1.2
:Homepage: https://github.com/ThomasGesseyJones/FullyBayesianForecastsExample
:Letter: TBD

.. image:: https://img.shields.io/badge/python-3.8-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ThomasGesseyJones/ErrorAffirmations/blob/main/LICENSE
   :alt: License information
.. image:: https://img.shields.io/badge/arXiv-2108.07282-b31b1b.svg?style=flat
    :target: https://arxiv.org/abs/2108.07282
    :alt: arXiv link


Example of a fully Bayesian forecast performed using an `Evidence Network <https://ui.adsabs.harvard.edu/abs/2023arXiv230511241J/abstract>`__.
This code also replicates the analysis of
`Gessey-Jones et al. (2023) <TBD>`__.
This repository thus serves the dual purposes of providing an example code base others
can modify to perform their own fully Bayesian forecasts and also providing a
reproducible analysis pipeline for the letter.

The overall goal of the code is to produce a fully Bayesian forecast of
the chance of a `REACH <https://ui.adsabs.harvard.edu/abs/2022NatAs...6..984D/abstract>`__-like experiment
making a significant detection of the 21-cm global signal, given a noise level. It also produces
figures showing how this conclusion changes with different astrophysical parameter values
and validates the forecast through blind coverage
tests and comparison to `PolyChord <https://ui.adsabs.harvard.edu/abs/2015MNRAS.453.4384H/abstract>`__.



Installation
------------

The repository is intended to be installed locally
by directly cloning it from GitHub. To do this run the following command in the terminal

.. code:: bash

    git clone git@github.com:ThomasGesseyJones/FullyBayesianForecastsExample.git

This will create a local copy of the repository. The pipeline can
then be run from the terminal (see below).


Structure and Usage
-------------------

The code is split into two main parts. The first part is the
modules which provide the general functionality of evidence networks,
data simulators, and prior samplers. The second part
is the scripts which run the fully Bayesian forecast.

There are three modules included in the repository:

- evidence_networks: This module contains the code for the evidence network
  class. This class is used to build the evidence network used in the forecasts.
  The module also provides an implementation of the l-POP exponential loss
  function.
  See the class docstring for more details of its capabilities and usage.
- priors: This module contains the code to generate functions that
  sample standard prior distributions. These include
  uniform, log-uniform, and Gaussian priors.
- simulators: This module defines simulators. In our code, these are functions
  that take a number of data simulations to run and return that number of mock data
  simulations alongside the values of any parameters that were used in the
  simulations. Submodules of this module define functions to generate specific
  simulators for models with noise only and models with a noisy 21-cm global signal.

These three modules are used in the three analysis scripts:

- verification_with_polychord.py: This script generates a range of mock data
  sets from both the noise-only model and the noisy-signal model, and then
  performs a Bayesian analysis on each of them.
  Evaluating the Bayes ratio between the two models of the data
  using Polychord. These results are then stored in the verification_data directory
  for later comparison with the results from the evidence network to
  verify its accuracy. It should be run first, ideally in parallel.
- train_evidence_network.py: This script builds the evidence network object and
  the data simulator functions, then trains the evidence network. Once trained
  it stores the evidence network in the models directory, then runs a blind
  coverage test on the network and validates its performance against the
  Polychord Bayes ratio evaluations from the previous script. It should
  be run second.
- visualize_forecasts.py: This script loads the evidence network from the
  models directory and uses it to forecast the chance of a REACH-like
  experiment detecting the 21-cm global signal by applying it to many
  data sets generated from the noisy-signal model. It then plots this result
  for fixed astrophysical parameters as in Figure 1 of the letter. This is
  done for detection significance thresholds of 2, 3 and 5 sigma. Selected
  numerical values are also output to a .txt file. It should be run last.


All three scripts have docstrings describing their role in more detail, as
well as giving advice on how to run them most efficiently. The
scripts can be run from the terminal using the following commands:

.. code:: bash

    python verification_with_polychord.py
    python train_evidence_network.py
    python visualize_forecasts.py

to run with the default noise level of 79 mK and replicate the
analysis from `Gessey-Jones et al. (2023) <TBD>`__.
Alternatively you can pass
the scripts a command line argument to specify the experiments noise level in K. For example
to run with a noise level of 100 mK you would run the following commands:

.. code:: bash

    python verification_with_polychord.py 0.1
    python train_evidence_network.py 0.1
    python visualize_forecasts.py 0.1

Two other files of interest are:

- fbf_utilities.py: which defines IO functions
  needed by the three scripts and a utility function to assemble the data
  simulators for the noise-only and noisy-signal model.
- configuration.yaml: which defines several parameters used in the code
  including the experimental frequency resolution, the priors on the
  astrophysical parameters of the global 21-cm signal model, and parameters
  that control which astrophysical parameters are plotted in the forecast
  figures. If you change the priors or resolution the entire pipeline
  needs to be rerun to get accurate results.

The various figures produced in the analysis are stored in the
figures_and_results directory alongside the timing_data to assess the
performance of the methodology and some summary statistics of the evidence
networks performance. The figures and data generated in the
analysis for `Gessey-Jones et al. (2023) <TBD>`__. are provided in this
repository for reference.

Licence and Citation
--------------------

The software is free to use on the MIT open source license.
If you use the software for academic purposes then we request that you cite
the `letter <TBD>`__ ::

  Gessey-Jones, T. and W. J. Handley. “Fully Bayesian Forecasts with Evidence
  Networks.” (2023). arXiv:2309.06942

If you are using Bibtex you can use the following to cite the letter

.. code:: bibtex

  @ARTICLE{GesseyJones2023,
           author = {{Gessey-Jones}, T. and {Handley}, W.~J.},
            title = "{Fully Bayesian Forecasts with Evidence Networks}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, General Relativity and Quantum Cosmology},
             year = 2023,
            month = sep,
              eid = {arXiv:2309.06942},
            pages = {arXiv:2309.06942},
              doi = {10.48550/arXiv.2309.06942},
    archivePrefix = {arXiv},
           eprint = {2309.06942},
     primaryClass = {astro-ph.IM}
  }

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


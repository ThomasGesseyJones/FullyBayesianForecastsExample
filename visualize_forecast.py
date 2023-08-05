"""Run and Visualize Fully Bayesian Forecast.

This script runs a fully Bayesian forecast for the detectability of the
global 21-cm signal and visualizes the results. The evidence network it
uses needs to be trained and saved first using
`train_evidence_network.py` or this script with raise an error.

If using this script it is recommended to run on a GPU for speed.
Plus some CPUs will not have enough memory for the mock data.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.025 K. Each different noise level
will require a different network to be trained.
"""


def main():
    """Run and visualize fully Bayesian forecast."""
    pass


if __name__ == "__main__":
    main()

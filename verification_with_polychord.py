"""Verify accuracy of evidence network with Polychord.

This script evaluates the Bayes ratio for a range of mock data sets
using Polychord. These values are then compared to the values previously
computed by the evidence network. The script outputs summary statistics
as well as plots of the Bayes ratio values.

This script is intended to be run after train_evidence_network.py has
been run to train the evidence network. If not run, the script will
fail.

It is recommended to run this script on a CPU since PolyChord does not
derive any benefit from GPUs.

The script can take an optional command line argument to specify the
noise sigma in K. The default is 0.025 K. This indicates the evidence
network to use for the comparison.
"""


def main():
    """Verify accuracy of evidence network with Polychord."""
    pass


if __name__ == "__main__":
    main()

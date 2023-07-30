"""Evidence Network.

This module contains the EvidenceNetwork class, which is used to represent
a neural network trained to predict the Bayes ratio between two models.
"""

# Required imports
from typing import Callable, Tuple, Optional
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras import backend as k_backend
import matplotlib.pyplot as plt


class EvidenceNetwork:
    """
    A neural network trained to predict the Bayes ratio between two models.

    Parameters
    ----------
    simulator_0: Callable
        A function that will simulate mock data using model 0.
        The function should take the number of data sets to simulate as input,
        and it should return the data as an array and any parameters of interest
        as a pandas dataframe.
    simulator_1: Callable
        A function that will simulate mock data using model 1
        The function should take the number of data sets to simulate as input,
        and it should return the data as an array and any parameters of interest
        as a pandas dataframe.

    Attributes
    ----------
    trained: bool
        Whether the network has been trained
    nn_model: keras.Model
        The neural network model
    training_data: np.ndarray
        The data used to train the network
    training_labels: np.ndarray
        The labels used to train the network
    validation_data: np.ndarray
        The data used to validate the network
    validation_labels: np.ndarray
        The labels used to validate the network
    """

    def __init__(self, simulator_0: Callable, simulator_1: Callable):
        """Initialize an EvidenceNetwork object.

        Parameters
        ----------
        simulator_0: Callable
            A function that will simulate mock data using model 0.
            The function should take the number of data sets to simulate as
            input, and it should return the data as an array and any parameters
            of interest as a pandas dataframe.
        simulator_1: Callable
            A function that will simulate mock data using model 1.
            The function should take the number of data sets to simulate as
            input, and it should return the data as an array and any parameters
            of interest as a pandas dataframe.
        """
        # Check models are compatible
        sample_data_0, _ = simulator_0(1)
        sample_data_1, _ = simulator_1(1)
        if sample_data_0.shape != sample_data_1.shape:
            raise ValueError("Mock data from both simulators must have the "
                             "same output shape.")
        if sample_data_0.dtype != sample_data_1.dtype:
            raise ValueError("Mock data from both simulators must have the "
                             "same type.")

        # Set attributes
        self.simulator_0 = simulator_0
        self.simulator_1 = simulator_1
        self._data_shape = sample_data_0.shape
        self.trained = False

        # Attributes to be defined later
        self.nn_model = None
        self.training_data = None
        self.training_labels = None
        self.validation_data = None
        self.validation_labels = None

    @staticmethod
    def default_nn_model(input_size: int) -> keras.Model:
        """Return a default neural network model.

        This is the model from the appendix of arXiv:2305.11241.

        Parameters
        ----------
        input_size: int
            The number of input features

        Returns
        -------
        keras.Model
            The default neural network model
        """
        inputs = layers.Input(shape=input_size)
        x = layers.Dense(130)(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16)(x)
        x = layers.LeakyReLU()(x)
        x_batch_norm_1 = layers.BatchNormalization()(x)  # Save for skip
        x = layers.Dense(16)(x_batch_norm_1)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Add()([x, x_batch_norm_1])  # Skip connection
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16)(x)
        x = layers.LeakyReLU()(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs,
                            name="jeffrey_wandelt_23_network")
        return model

    def get_simulated_data(self,
                           num_samples_per_model: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a mix of simulated data and labels from both models.

        Parameters
        ----------
        num_samples_per_model: int
            The number of samples to draw from each model

        Returns
        -------
        data_sample: np.ndarray
            The simulated data, 50-50 mix of both models
        labels_sample: np.ndarray
            The labels, 0 for model 0, 1 for model 1
        """
        # Generate data from both models
        sample_data_0, _ = self.simulator_0(num_samples_per_model)
        labels_data_0 = np.zeros(num_samples_per_model, dtype=float)

        sample_data_1, _ = self.simulator_1(num_samples_per_model)
        labels_data_1 = np.ones(num_samples_per_model, dtype=float)

        # Combine data in randomized order
        data_sample = np.concatenate((sample_data_0, sample_data_1))
        labels_sample = np.concatenate((labels_data_0, labels_data_1))
        random_indices = np.random.permutation(
            np.arange(2 * num_samples_per_model))
        data_sample = data_sample[random_indices, ...]
        labels_sample = labels_sample[random_indices]
        return data_sample, labels_sample

    def train(self,
              nn_model: keras.Model = None,
              train_data_samples_per_model: int = 1_000_000,
              validation_data_samples_per_model: int = 200_000,
              epochs: int = 10,
              batch_size: int = 100) -> None:
        """Train the Bayes ratio network.

        Parameters
        ----------
        nn_model: keras.Model
            The neural network model to train. If None, a default model
            from arXiv:2305.11241 appendix is used.
        train_data_samples_per_model: int, default=1_000_000
            The number of data samples to simulate from each model for training
        validation_data_samples_per_model: int, default=200_000
            The number of data samples to simulate from each model for
            validation
        epochs: int, default=10
            The number of epochs to train for
        batch_size: int, default=100
            The batch size to use for training
        """
        # Set-up NN, default from arXiv:2305.11241 appendix if not given
        if nn_model is None:
            nn_model = self.default_nn_model(self._data_shape)
        self.nn_model = nn_model

        # Compile model, using details from arXiv:2305.11241
        self.nn_model.compile(
            loss=l_pop_exponential_loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=ExponentialDecay(
                    initial_learning_rate=1e-4,
                    decay_steps=1000,
                    decay_rate=0.95,
                )),
            metrics=["accuracy"],
        )

        # Create simulated data from both models for training and validation
        sample_data, labels_data = \
            self.get_simulated_data(train_data_samples_per_model)
        validation_sample_data, validation_labels_data = \
            self.get_simulated_data(validation_data_samples_per_model)

        # Store simulated data in case useful later on
        self.training_data = sample_data
        self.training_labels = labels_data
        self.validation_data = validation_sample_data
        self.validation_labels = validation_labels_data

        # Train model without early stopping
        self.nn_model.fit(sample_data, labels_data,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=2,
                          validation_data=(validation_sample_data,
                                           validation_labels_data))
        self.trained = True
        return

    def evaluate_log_bayes_ratio(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the log Bayes ratio between model 1 and 0.

        Parameters
        ----------
        data: np.ndarray
            A dataset or batch of data sets to feed to the network.

        Returns
        -------
        log_bayes_ratio: np.ndarray
            The log Bayes ratio for the data set(s)
        """
        if not self.trained:
            raise ValueError("Network has not been trained yet.")

        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        nn_output = self.nn_model(tf.constant(data), training=False)
        return leaky_parity_odd_transformation(nn_output)

    def evaluate_bayes_ratio(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the Bayes ratio between model 1 and 0.

        Parameters
        ----------
        data: np.ndarray
            A dataset or batch of data sets to feed to the network.

        Returns
        -------
        bayes_ratio: np.ndarray
            The Bayes ratio for the data set(s)
        """
        if not self.trained:
            raise ValueError("Network has not been trained yet.")

        return np.exp(self.evaluate_log_bayes_ratio(data))

    def save(self, filename: str):
        """Save the network to file.

        Parameters
        ----------
        filename: str
            The filename to save the network to.
        """
        self.nn_model.save(filename)

    def load(self, filename: str):
        """Load the network from file.

        Parameters
        ----------
        filename: str
            The filename to load the network from.
        """
        self.nn_model = keras.models.load_model(filename)
        self.trained = True

    def blind_coverage_test(self,
                            num_validation_samples: int = 2_000,
                            num_probability_bins: int = 50,
                            min_in_bin: int = 5,
                            plotting_ax: Optional[plt.Axes] = None,
                            plot_errors: bool = True,
                            default_plot_formatting: bool = True,
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a blind coverage test on the network.

        Parameters
        ----------
        num_validation_samples: int, default=2_000
            The number of validation samples to use for the test
        num_probability_bins: int, default=50
            The number of probability bins to use
        min_in_bin: int, default=5
            The minimum number of samples in a bin for it to be used in the
            test. If a bin has fewer samples than this, it is set to NaN.
        plotting_ax: Optional[plt.Axes], default=None
            An optional matplotlib axes to plot the results on
        plot_errors: bool, default=True
            Whether to plot the error bars on the testing plot, only used if
            plotting_ax is not None
        default_plot_formatting: bool, default=True
            Whether to apply default formatting to the plot, only used if
            plotting_ax is not None

        Returns
        -------
        posterior_bin_edges: np.ndarray
            The model posterior probability bin edges used for the
            blind coverage test
        bin_model_1_proportions: np.ndarray
            The proportion of samples in each bin from model 1
        """
        # Setup bin edges
        posterior_bin_edges = np.linspace(0, 1, num_probability_bins + 1)
        posterior_bin_midpoints = (posterior_bin_edges[1:] +
                                   posterior_bin_edges[:-1]) / 2

        # Generate validation data
        validation_data, validation_labels = \
            self.get_simulated_data(num_validation_samples)

        # Evaluate Bayes ratio
        bayes_ratio = self.evaluate_bayes_ratio(validation_data)
        model_1_posterior = bayes_ratio / (1 + bayes_ratio)

        # Bin data
        num_model_1_in_bin = np.zeros(num_probability_bins)
        num_model_0_in_bin = np.zeros(num_probability_bins)
        for posterior, label in zip(model_1_posterior, validation_labels):
            bin_index = np.searchsorted(posterior_bin_edges[1:], posterior)
            if bin_index == num_probability_bins:
                bin_index -= 1
            if label == 1:
                num_model_1_in_bin[bin_index] += 1
            else:
                num_model_0_in_bin[bin_index] += 1

        # Calculate bin proportions
        bin_model_1_proportions = num_model_1_in_bin / (
                num_model_1_in_bin + num_model_0_in_bin)

        # Set bins with too few samples to NaN
        bin_model_1_proportions[
            num_model_1_in_bin + num_model_0_in_bin < min_in_bin] = np.nan

        # Optionally plot results
        if plotting_ax is None:
            return posterior_bin_edges, bin_model_1_proportions

        if default_plot_formatting:
            # Formatting
            plotting_ax.set_xlabel("Posterior Model 1 Probability")
            plotting_ax.set_ylabel("Proportion of Model 1 Samples")
            plotting_ax.set_xlim(0, 1)
            plotting_ax.set_ylim(0, 1)
            plotting_ax.set_xticks([0, 0.5, 1])
            plotting_ax.set_yticks([0, 0.5, 1])
            plotting_ax.grid(True)
            plotting_ax.set_axisbelow(True)

            # Expectation line
            plotting_ax.plot([0, 1], [0, 1], color="black", linestyle="--")

            # Formatting for error bars / scatter plot
            plotting_kwargs = {"marker": "x", "color": "black"}
        else:
            plotting_kwargs = {}

        if plot_errors:
            plotting_ax.errorbar(
                x=posterior_bin_midpoints,
                y=bin_model_1_proportions,
                # For error formula see
                # https://www2.sjsu.edu/faculty/gerstman/StatPrime
                # on Standard Error of a Proportion
                yerr=np.sqrt(bin_model_1_proportions
                             * (1 - bin_model_1_proportions)
                             / (num_model_1_in_bin + num_model_0_in_bin)),
                **plotting_kwargs)
        else:
            plotting_ax.scatter(
                x=posterior_bin_midpoints,
                y=bin_model_1_proportions,
                **plotting_kwargs)

        return posterior_bin_edges, bin_model_1_proportions


def leaky_parity_odd_transformation(x: np.ndarray,
                                    alpha: float = 2.0) -> np.ndarray:
    """Apply the leaky parity-odd transformation to a value.

    Parameters
    ----------
    x: np.ndarray
        A value to transform
    alpha: float, default=2.0
        The exponent of the transformation

    Returns
    -------
    x_transformed: np.ndarray
        The transformed value
    """
    return x * k_backend.pow(k_backend.abs(x), alpha - 1) + x


def l_pop_exponential_loss(model_label: float, f_x: np.ndarray) -> np.ndarray:
    """l-POP-Exponential loss function from arxiv:2305.11241.

    Parameters
    ----------
    model_label: float
        The true value of the model label (either 0.0 or 1.0)
    f_x: np.ndarray
        The value output by network

    Returns
    -------
    loss: np.ndarray
        The loss function value
    """
    return k_backend.exp((0.5 - model_label) *
                         leaky_parity_odd_transformation(f_x))

"""
Simple Perceptron Classifier Implementation
Using only Python built-in functions and math library
"""

import random
import math


class Perceptron:
    """
    A simple perceptron classifier for binary classification.

    The perceptron learns a linear decision boundary to separate two classes.
    It uses the perceptron learning algorithm with a step activation function.
    """

    def __init__(self, learning_rate=0.01, epochs=100, random_seed=None):
        """
        Initialize the perceptron.

        Args:
            learning_rate (float): Learning rate for weight updates
            epochs (int): Number of training iterations
            random_seed (int): Seed for random weight initialization
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.random_seed = random_seed

    def _initialize_weights(self, n_features):
        """Initialize weights randomly."""
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Initialize weights with small random values
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = random.uniform(-0.1, 0.1)

    def _activation(self, x):
        """
        Step activation function.
        Returns 1 if x >= 0, else 0.
        """
        return 1 if x >= 0 else 0

    def _predict_single(self, x):
        """
        Make prediction for a single sample.

        Args:
            x (list): Feature vector

        Returns:
            int: Predicted class (0 or 1)
        """
        # Calculate weighted sum: w·x + b
        weighted_sum = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._activation(weighted_sum)

    def fit(self, X, y):
        """
        Train the perceptron on the training data.

        Args:
            X (list of lists): Training features, shape (n_samples, n_features)
            y (list): Target labels (0 or 1), shape (n_samples,)

        Returns:
            self: Returns the trained perceptron
        """
        n_samples = len(X)
        n_features = len(X[0])

        # Initialize weights
        self._initialize_weights(n_features)

        # Training loop
        for epoch in range(self.epochs):
            errors = 0

            # Iterate through all training samples
            for xi, yi in zip(X, y):
                # Make prediction
                prediction = self._predict_single(xi)

                # Calculate error
                error = yi - prediction

                # Update weights if there's an error
                if error != 0:
                    errors += 1
                    # Update rule: w = w + learning_rate * error * x
                    self.weights = [
                        w + self.learning_rate * error * xi_val
                        for w, xi_val in zip(self.weights, xi)
                    ]
                    # Update bias: b = b + learning_rate * error
                    self.bias += self.learning_rate * error

            # Optional: print progress (commented out by default)
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch {epoch + 1}/{self.epochs}, Errors: {errors}")

            # Early stopping if no errors
            if errors == 0:
                break

        return self

    def predict(self, X):
        """
        Make predictions for multiple samples.

        Args:
            X (list of lists): Feature vectors, shape (n_samples, n_features)

        Returns:
            list: Predicted classes (0 or 1) for each sample
        """
        if self.weights is None:
            raise ValueError("Perceptron must be trained before making predictions")

        return [self._predict_single(xi) for xi in X]

    def score(self, X, y):
        """
        Calculate accuracy on the given data.

        Args:
            X (list of lists): Feature vectors
            y (list): True labels

        Returns:
            float: Accuracy (proportion of correct predictions)
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)

    def get_weights(self):
        """
        Get the learned weights and bias.

        Returns:
            tuple: (weights, bias)
        """
        return self.weights, self.bias


def dot_product(a, b):
    """Calculate dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def vector_add(a, b):
    """Add two vectors element-wise."""
    return [ai + bi for ai, bi in zip(a, b)]


def scalar_multiply(scalar, vector):
    """Multiply a vector by a scalar."""
    return [scalar * vi for vi in vector]

"""
Unit tests for the Perceptron classifier
Focus on simple two-group classification problems
"""

import unittest
from perceptron import Perceptron


class TestPerceptronTwoGroups(unittest.TestCase):
    """Test perceptron on simple two-group classification problems"""

    def test_horizontal_separation(self):
        """Test: Two groups separated horizontally (by x-axis)"""
        # Group 0: left side (x < 3)
        # Group 1: right side (x > 5)
        X = [
            [1, 5], [2, 4], [1, 6], [2, 5],  # Group 0
            [6, 5], [7, 4], [6, 6], [8, 5]   # Group 1
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        model = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should perfectly separate horizontal groups")

    def test_vertical_separation(self):
        """Test: Two groups separated vertically (by y-axis)"""
        # Group 0: bottom (y < 3)
        # Group 1: top (y > 5)
        X = [
            [5, 1], [4, 2], [6, 1], [5, 2],  # Group 0
            [5, 6], [4, 7], [6, 6], [5, 8]   # Group 1
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        model = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should perfectly separate vertical groups")

    def test_diagonal_separation(self):
        """Test: Two groups separated diagonally"""
        # Group 0: bottom-left
        # Group 1: top-right
        X = [
            [1, 1], [2, 2], [1, 2], [2, 1],  # Group 0
            [7, 7], [8, 8], [7, 8], [8, 7]   # Group 1
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        model = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should perfectly separate diagonal groups")

    def test_clustered_groups(self):
        """Test: Two well-separated clustered groups"""
        # Group 0: cluster around (2, 2)
        # Group 1: cluster around (8, 8)
        X = [
            [2, 2], [2.5, 2.5], [1.5, 2], [2, 1.5],    # Group 0
            [8, 8], [8.5, 8.5], [7.5, 8], [8, 7.5]     # Group 1
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        model = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should perfectly separate clustered groups")


class TestPerceptronLogicalGates(unittest.TestCase):
    """Test perceptron on simple logical gate problems"""

    def test_and_gate(self):
        """Test: AND gate (linearly separable)"""
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 0, 0, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions, y, "Should learn AND gate perfectly")

    def test_or_gate(self):
        """Test: OR gate (linearly separable)"""
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions, y, "Should learn OR gate perfectly")

    def test_nand_gate(self):
        """Test: NAND gate (linearly separable)"""
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [1, 1, 1, 0]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions, y, "Should learn NAND gate perfectly")


class TestPerceptronBasicFunctionality(unittest.TestCase):
    """Test basic perceptron functionality"""

    def test_fit_returns_self(self):
        """Test: fit() should return self for method chaining"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=10)
        result = model.fit(X, y)

        self.assertIs(result, model, "fit() should return self")

    def test_predict_before_fit_raises_error(self):
        """Test: predict() before fit() should raise ValueError"""
        X = [[1, 2], [2, 3]]

        model = Perceptron()

        with self.assertRaises(ValueError):
            model.predict(X)

    def test_weights_initialized_after_fit(self):
        """Test: Weights should be initialized after training"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=10)
        self.assertIsNone(model.weights, "Weights should be None before training")

        model.fit(X, y)

        self.assertIsNotNone(model.weights, "Weights should be set after training")
        self.assertEqual(len(model.weights), 2, "Should have 2 weights for 2 features")

    def test_get_weights(self):
        """Test: get_weights() returns correct format"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=10, random_seed=42)
        model.fit(X, y)

        weights, bias = model.get_weights()

        self.assertIsInstance(weights, list, "Weights should be a list")
        self.assertIsInstance(bias, float, "Bias should be a float")
        self.assertEqual(len(weights), 2, "Should have 2 weights for 2 features")

    def test_score_accuracy(self):
        """Test: score() returns correct accuracy"""
        X = [[1, 1], [2, 2], [5, 5], [6, 6]]
        y = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)

        self.assertIsInstance(accuracy, float, "Accuracy should be a float")
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be <= 1")

    def test_reproducibility_with_seed(self):
        """Test: Same seed should produce same results"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y = [0, 0, 1, 1]

        model1 = Perceptron(learning_rate=0.1, epochs=10, random_seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = Perceptron(learning_rate=0.1, epochs=10, random_seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        self.assertEqual(pred1, pred2, "Same seed should produce same predictions")


class TestPerceptronEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_single_feature(self):
        """Test: Perceptron with single feature"""
        # Group 0: small values
        # Group 1: large values
        X = [[1], [2], [3], [7], [8], [9]]
        y = [0, 0, 0, 1, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should separate groups with single feature")

    def test_three_features(self):
        """Test: Perceptron with three features"""
        # Group 0: points near origin
        # Group 1: points far from origin
        X = [
            [1, 1, 1], [2, 1, 1], [1, 2, 1],      # Group 0
            [8, 8, 8], [9, 8, 8], [8, 9, 8]       # Group 1
        ]
        y = [0, 0, 0, 1, 1, 1]

        model = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertGreater(accuracy, 0.8, "Should handle 3D data")

    def test_all_same_label(self):
        """Test: All samples have the same label"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5]]
        y = [1, 1, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=10, random_seed=42)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions, y, "Should predict all same label")

    def test_minimal_dataset(self):
        """Test: Minimal dataset (2 samples)"""
        X = [[1, 1], [5, 5]]
        y = [0, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions, y, "Should handle minimal dataset")


class TestPerceptronPredictionConsistency(unittest.TestCase):
    """Test prediction consistency and generalization"""

    def test_prediction_on_training_data(self):
        """Test: Predictions should be accurate on training data"""
        X = [[1, 1], [2, 2], [6, 6], [7, 7]]
        y = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X, y)

        accuracy = model.score(X, y)
        self.assertEqual(accuracy, 1.0, "Should perfectly fit training data")

    def test_prediction_on_new_data(self):
        """Test: Should reasonably predict on new similar data"""
        X_train = [[1, 1], [2, 2], [6, 6], [7, 7]]
        y_train = [0, 0, 1, 1]

        model = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
        model.fit(X_train, y_train)

        # Test data similar to training data
        X_test = [[1.5, 1.5], [6.5, 6.5]]
        y_test = [0, 1]

        predictions = model.predict(X_test)
        self.assertEqual(predictions, y_test, "Should generalize to similar new data")


def run_tests_verbose():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPerceptronTwoGroups))
    suite.addTests(loader.loadTestsFromTestCase(TestPerceptronLogicalGates))
    suite.addTests(loader.loadTestsFromTestCase(TestPerceptronBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestPerceptronEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerceptronPredictionConsistency))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("PERCEPTRON CLASSIFIER - TWO GROUP CLASSIFICATION TESTS")
    print("=" * 70)
    print()

    result = run_tests_verbose()

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

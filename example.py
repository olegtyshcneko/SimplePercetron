"""
Example usage of the Perceptron classifier
"""

from perceptron import Perceptron


def example_1_simple_and_gate():
    """Example 1: Learning the AND gate"""
    print("=" * 50)
    print("Example 1: AND Gate")
    print("=" * 50)

    # Training data for AND gate
    # Features: [x1, x2], Label: x1 AND x2
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y_train = [0, 0, 0, 1]  # AND gate output

    # Create and train perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
    perceptron.fit(X_train, y_train)

    # Make predictions
    predictions = perceptron.predict(X_train)

    # Display results
    print("\nInput -> Prediction (Expected)")
    for x, pred, expected in zip(X_train, predictions, y_train):
        print(f"{x} -> {pred} ({expected})")

    # Calculate accuracy
    accuracy = perceptron.score(X_train, y_train)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Show learned weights
    weights, bias = perceptron.get_weights()
    print(f"Learned weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Learned bias: {bias:.4f}")


def example_2_or_gate():
    """Example 2: Learning the OR gate"""
    print("\n" + "=" * 50)
    print("Example 2: OR Gate")
    print("=" * 50)

    # Training data for OR gate
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y_train = [0, 1, 1, 1]  # OR gate output

    # Create and train perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=100, random_seed=42)
    perceptron.fit(X_train, y_train)

    # Make predictions
    predictions = perceptron.predict(X_train)

    # Display results
    print("\nInput -> Prediction (Expected)")
    for x, pred, expected in zip(X_train, predictions, y_train):
        print(f"{x} -> {pred} ({expected})")

    # Calculate accuracy
    accuracy = perceptron.score(X_train, y_train)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")


def example_3_linearly_separable_data():
    """Example 3: 2D linearly separable classification"""
    print("\n" + "=" * 50)
    print("Example 3: 2D Linearly Separable Data")
    print("=" * 50)

    # Create a simple 2D dataset
    # Class 0: points in bottom-left region
    # Class 1: points in top-right region
    X_train = [
        [1.0, 1.5],
        [2.0, 1.8],
        [1.5, 2.0],
        [2.5, 2.2],
        [5.0, 5.5],
        [6.0, 6.2],
        [5.5, 6.0],
        [6.5, 6.8]
    ]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    # Create and train perceptron
    perceptron = Perceptron(learning_rate=0.01, epochs=100, random_seed=42)
    perceptron.fit(X_train, y_train)

    # Test data
    X_test = [
        [1.8, 2.1],  # Should be class 0
        [6.2, 6.5],  # Should be class 1
        [2.0, 2.0],  # Should be class 0
        [5.5, 5.8],  # Should be class 1
    ]

    predictions = perceptron.predict(X_test)

    print("\nTest predictions:")
    for x, pred in zip(X_test, predictions):
        print(f"{x} -> Class {pred}")

    # Calculate training accuracy
    train_accuracy = perceptron.score(X_train, y_train)
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")

    # Show learned weights
    weights, bias = perceptron.get_weights()
    print(f"Learned weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Learned bias: {bias:.4f}")


def example_4_manual_test():
    """Example 4: Manual step-by-step demonstration"""
    print("\n" + "=" * 50)
    print("Example 4: Manual Step-by-Step")
    print("=" * 50)

    # Simple dataset
    X = [[2, 3], [1, 1], [4, 5], [3, 2]]
    y = [0, 0, 1, 1]

    print("Training data:")
    for xi, yi in zip(X, y):
        print(f"  Features: {xi}, Label: {yi}")

    # Train perceptron
    perceptron = Perceptron(learning_rate=0.1, epochs=50, random_seed=123)
    print("\nTraining perceptron...")
    perceptron.fit(X, y)

    # Make predictions
    print("\nPredictions on training data:")
    predictions = perceptron.predict(X)
    for xi, pred, true in zip(X, predictions, y):
        status = "✓" if pred == true else "✗"
        print(f"  {xi} -> Predicted: {pred}, Actual: {true} {status}")

    # Test on new data
    X_new = [[2.5, 3.5], [3.5, 3.0]]
    print("\nPredictions on new data:")
    new_predictions = perceptron.predict(X_new)
    for xi, pred in zip(X_new, new_predictions):
        print(f"  {xi} -> Predicted: {pred}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PERCEPTRON CLASSIFIER EXAMPLES")
    print("=" * 50)

    # Run all examples
    example_1_simple_and_gate()
    example_2_or_gate()
    example_3_linearly_separable_data()
    example_4_manual_test()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50 + "\n")

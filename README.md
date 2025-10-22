# Simple Perceptron Classifier

A simple implementation of the Perceptron algorithm for binary classification, using only Python's built-in libraries (no NumPy, scikit-learn, or other external dependencies).

## What is a Perceptron?

A perceptron is the simplest form of a neural network - a single-layer linear classifier. It learns to classify data into two categories by finding a linear decision boundary (a line in 2D, a plane in 3D, or a hyperplane in higher dimensions).

## Algorithm

The perceptron learning algorithm works as follows:

1. **Initialize** weights randomly and bias to small values
2. For each training example:
   - **Calculate** the weighted sum: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
   - **Predict** class: `y = 1 if z ≥ 0 else 0`
   - **Update** weights if prediction is wrong:
     - `w = w + learning_rate × (y_true - y_pred) × x`
     - `b = b + learning_rate × (y_true - y_pred)`
3. **Repeat** for multiple epochs until convergence

## Features

- Pure Python implementation (no external dependencies except `random` and `math`)
- Simple, readable code
- Suitable for linearly separable binary classification problems
- Includes multiple examples demonstrating usage

## Files

- `perceptron.py` - Main Perceptron class implementation
- `example.py` - Demonstration examples (AND gate, OR gate, 2D classification)
- `README.md` - This file

## Usage

### Basic Example

```python
from perceptron import Perceptron

# Training data (AND gate)
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]

# Create and train perceptron
model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
print(predictions)  # [0, 0, 0, 1]

# Check accuracy
accuracy = model.score(X_train, y_train)
print(f"Accuracy: {accuracy * 100}%")
```

### Running Examples

```bash
python example.py
```

This will run several examples:
1. AND gate classification
2. OR gate classification
3. 2D linearly separable data
4. Manual step-by-step demonstration

## Class API

### Perceptron

#### Constructor
```python
Perceptron(learning_rate=0.01, epochs=100, random_seed=None)
```

**Parameters:**
- `learning_rate` (float): Step size for weight updates (default: 0.01)
- `epochs` (int): Number of training iterations (default: 100)
- `random_seed` (int): Seed for reproducible results (default: None)

#### Methods

**fit(X, y)**
- Train the perceptron on data
- `X`: List of feature vectors (2D list)
- `y`: List of labels (0 or 1)
- Returns: self

**predict(X)**
- Make predictions for new data
- `X`: List of feature vectors
- Returns: List of predicted labels (0 or 1)

**score(X, y)**
- Calculate accuracy
- `X`: Feature vectors
- `y`: True labels
- Returns: Accuracy (float between 0 and 1)

**get_weights()**
- Get learned parameters
- Returns: Tuple of (weights, bias)

## Limitations

1. **Linearly separable data only**: The perceptron can only learn linearly separable patterns. It cannot learn XOR or other non-linearly separable functions.

2. **Binary classification only**: This implementation only handles two classes (0 and 1).

3. **No normalization**: For best results, normalize your features before training.

## Mathematical Details

The perceptron implements:

**Decision function**: f(x) = sign(w·x + b)

**Update rule**:
- w ← w + η(y - ŷ)x
- b ← b + η(y - ŷ)

Where:
- w = weight vector
- b = bias
- η = learning rate
- y = true label
- ŷ = predicted label
- x = input features

## License

This is a simple educational implementation. Feel free to use and modify as needed.

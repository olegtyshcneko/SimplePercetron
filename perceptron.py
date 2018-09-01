from numpy.random import normal
from numpy import dot, sign


def get_starting_weights(features_count):
    return normal(0, 0.1, features_count + 1)


def do_learning_step(X, y, w, learning_rate=0.1):
    """
    Does perceptron learning step, updating weights
    according to prediction results
    :param X: array of features
    :param y: target values array(y_real)
    :param learning_rate: learning rate
    :param w: array of weights
    :return: new updated weights
    """
    errors_count = 0
    for (i, x) in X.iterrows():
        prediction = predict_single(x, w)
        real = y[i]
        real_prediction_diff = real - prediction
        update = learning_rate * real_prediction_diff
        w[0] += update
        w[1:] = [j + (update * x[i]) for (i, j) in enumerate(w[1:])]
        errors_count += 1 if real != prediction else 0

    return w, errors_count


def predict_single(x, w):
    """
    Does perceptron prediction on single object
    :param x: array of object features
    :param w: array of weights
    :return: binary prediction
    """
    bias_weight = w[0]
    weights = w[1:]
    net_sum = weights.dot(x)
    s = sign(bias_weight + net_sum)
    return s

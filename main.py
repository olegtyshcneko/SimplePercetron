from data_iris import get_iris_training_set, get_iris_test_set
import perceptron as pc

learning_steps = 50
#
iris_training_set_data, iris_training_set_targets = get_iris_training_set()
iris_test_set_data, iris_test_set_targets = get_iris_test_set()
#
features_count = iris_test_set_data.shape[1]
weights = pc.get_starting_weights(features_count)
epochs = []

for i in range(learning_steps):
    weights, errors_count = pc.do_learning_step(iris_training_set_data, iris_training_set_targets, weights)
    epochs.append(errors_count)

# predicted_test_data = list(map(lambda td: pc.predict_single(td, weights), iris_test_set_data))
iris_test_set_data
# predictions_compared = [1 if iris_test_set_targets[i] == j else 0 for (i,j) in enumerate(predicted_test_data)]
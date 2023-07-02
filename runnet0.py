import numpy as np

class Agent:
    def __init__(self, network):
        self.neural_network = network
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, dropout_rate=0.2):
        self.weights = [
            np.random.randn(hidden1_dim, input_dim) / np.sqrt(input_dim),
            np.random.randn(hidden2_dim, hidden1_dim) / np.sqrt(hidden1_dim),
            np.random.randn(hidden3_dim, hidden2_dim) / np.sqrt(hidden2_dim),
            np.random.randn(output_dim, hidden3_dim) / np.sqrt(hidden3_dim),
        ]

        self.biases = [
            np.random.randn(hidden1_dim, 1),
            np.random.randn(hidden2_dim, 1),
            np.random.randn(hidden3_dim, 1),
            np.random.randn(output_dim, 1)
        ]
        self.dropout_rate = dropout_rate
        # New instance variables for batch norm
        self.bn_means = [np.zeros((1, dim)) for dim in [hidden1_dim, hidden2_dim, hidden3_dim, output_dim]]
        self.bn_vars = [np.zeros((1, dim)) for dim in [hidden1_dim, hidden2_dim, hidden3_dim, output_dim]]
        self.bn_decay = 0.9  # Decay rate for the running averages

    def propagate(self, X, training=True, return_hidden=False):
        X = np.array(X).reshape(-1, self.weights[0].shape[1])  # Ensure X has the correct shape
        hidden1 = self.elu(np.dot(X, self.weights[0].T) + self.biases[0].T)

        hidden2 = self.relu(np.dot(hidden1, self.weights[1].T) + self.biases[1].T)
        hidden2 = self.batch_norm(hidden2, 1, training)  # Specify layer index and training

        hidden3 = self.elu(np.dot(hidden2, self.weights[2].T) + self.biases[2].T)
        hidden3 = self.dropout(hidden3, self.dropout_rate, training)

        output_layer = self.sigmoid(np.dot(hidden3, self.weights[3].T) + self.biases[3].T)

        # If we're interested in the activations of the last hidden layer for draw TSNE:
        if return_hidden:
            return hidden3
        else:
            return output_layer

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def elu(self, x,
            alpha=1.0):  # The alpha parameter controls the value that ELU converges towards for negative net inputs
        return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1))

    def dropout(self, X, dropout_rate, training=True):
        if not training:
            return X
        keep_prob = 1 - dropout_rate
        mask = np.random.binomial(1, keep_prob, size=X.shape) / keep_prob
        return X * mask

    def batch_norm(self, X, layer, training=True):
        if training:
            mean = np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)

            # Update running averages
            self.bn_means[layer] = self.bn_decay * self.bn_means[layer] + (1 - self.bn_decay) * mean
            self.bn_vars[layer] = self.bn_decay * self.bn_vars[layer] + (1 - self.bn_decay) * var
        else:
            # Use running averages
            mean = self.bn_means[layer]
            var = self.bn_vars[layer]

        X_norm = (X - mean) / np.sqrt(var + 1e-8)
        return X_norm


def load_network(filename):
    # Load the dictionary from the numpy .npz file
    network_dict = np.load(filename)

    # Extract the architecture information
    input_dim = network_dict['input_dim'].item()
    hidden1_dim = network_dict['hidden1_dim'].item()
    hidden2_dim = network_dict['hidden2_dim'].item()
    hidden3_dim = network_dict['hidden3_dim'].item()
    output_dim = network_dict['output_dim'].item()
    dropout_rate = network_dict['dropout_rate'].item()

    # Create a new network with the loaded architecture
    network = NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, dropout_rate)

    # Extract the weights and biases
    weights = [network_dict[f'weight_{i}'] for i in range(len(network.weights))]
    biases = [network_dict[f'bias_{i}'] for i in range(len(network.biases))]

    # Assign the loaded weights and biases to the network
    network.weights = weights
    network.biases = biases

    return network


def predict(network, X):
    predictions = network.propagate(X, training=False)
    predicted_labels = np.round(predictions)
    return predicted_labels

def save_predictions(predictions, filename):
    np.savetxt(filename, predictions, fmt='%d')

def runnet(network_file, data_file, output_file):
    # Load the network from the file
    network = load_network(network_file)

    # Load the unlabeled data
    with open(data_file, "r") as f:
        data = f.readlines()

    X = []
    for line in data:
        split_line = line.split()
        X.append([int(char) for char in split_line[0]])

    X = np.array(X)
    print(f"X.shape[0] = {X.shape[1]}")
    # Predict the labels of the unlabeled data
    predictions = predict(network, X)

    # Save the predicted labels to a file
    save_predictions(predictions, output_file)


def compare_files(true_labels_file, predictions_file):
    with open(true_labels_file, 'r') as file1, open(predictions_file, 'r') as file2:
        true_labels = file1.readlines()
        predictions = file2.readlines()

    correct = 0
    incorrect = 0

    for true_label, prediction in zip(true_labels, predictions):
        if true_label.strip() == prediction.strip():
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (correct + incorrect) * 100  # Calculating accuracy
    return correct, incorrect, accuracy



if __name__ == "__main__":
    true_labels_file = "true_label0_MoreV1.txt"
    predictions_file = "predicted_labels0_MoreV1.txt"
    test_data_file = "testnet0_MoreV1.txt"
    runnet('wnet0.npz', test_data_file, predictions_file)
    print(f"Done")
    # Use the function

    correct, incorrect, accuracy = compare_files(true_labels_file, predictions_file)

    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")

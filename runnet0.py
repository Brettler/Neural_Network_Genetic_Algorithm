import numpy as np

class Agent:
    def __init__(self, network):
        self.neural_network = network
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, dropout_rate=0.2):
        self.weights = [
            np.random.randn(hidden1_dim, input_dim),
            np.random.randn(hidden2_dim, hidden1_dim),
            np.random.randn(hidden3_dim, hidden2_dim),
            np.random.randn(output_dim, hidden3_dim),

        ]
        self.biases = [
            np.random.randn(hidden1_dim, 1),
            np.random.randn(hidden2_dim, 1),
            np.random.randn(hidden3_dim, 1),
            np.random.randn(output_dim, 1)
        ]
        self.dropout_rate = dropout_rate

    def propagate(self, X, training=True):
        X = np.array(X).reshape(-1, self.weights[0].shape[1])  # Ensure X has the correct shape
        hidden1 = self.relu(np.dot(X, self.weights[0].T) + self.biases[0].T)
        #
        # if training:
        #     self.dropout_rate = 0.2
        #     # Apply dropout to hidden1
        #     mask1 = np.random.binomial(1, 1 - self.dropout_rate, size=hidden1.shape) / (1 - self.dropout_rate)
        #     hidden1 *= mask1

        hidden2 = self.relu(np.dot(hidden1, self.weights[1].T) + self.biases[1].T)

        # if training:
        #     # Apply dropout to hidden layers
        #     mask2 = np.random.binomial(1, 1 - self.dropout_rate, size=hidden2.shape) / (1 - self.dropout_rate)
        #     hidden2 *= mask2

        hidden3 = self.relu(np.dot(hidden2, self.weights[2].T) + self.biases[2].T)


        output_layer = self.sigmoid(np.dot(hidden3, self.weights[3].T) + self.biases[3].T)

        return output_layer


    def relu(self, x):
        return np.maximum(0, x)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def save_network(agent, filename):
    with open(filename, "w") as f:
        # Saving dimensions
        f.write(f"{len(agent.neural_network.weights)}\n") # number of layers
        for weight in agent.neural_network.weights:
            f.write(f"{weight.shape[0]} {weight.shape[1]}\n") # dimensions of each layer

        # Saving weights
        for weight in agent.neural_network.weights:
            np.savetxt(f, weight)

        # Saving biases
        for bias in agent.neural_network.biases:
            np.savetxt(f, bias)

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
    runnet('wnet.npz', test_data_file, predictions_file)
    print(f"Done")
    # Use the function

    correct, incorrect, accuracy = compare_files(true_labels_file, predictions_file)

    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")

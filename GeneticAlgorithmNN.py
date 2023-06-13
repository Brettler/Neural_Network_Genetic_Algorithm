import random
import numpy as np

# Function to initialize network
def initialize_network(n_inputs, n_hidden_layers, n_neurons, n_outputs):
    network = list()
    for _ in range(n_hidden_layers):
        hidden_layer = [{'weights': np.random.uniform(-1.0, 1.0, n_inputs + 1)} for _ in range(n_neurons)]
        network.append(hidden_layer)
    output_layer = [{'weights': np.random.uniform(-1.0, 1.0, n_neurons + 1)} for _ in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights, inputs):
    inputs = np.array(inputs)
    print(weights.shape)  # for debugging
    print(inputs.shape)  # for debugging
    activation = weights[-1] + np.dot(weights[:-1], inputs)
    return activation

def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))

def transfer_derivative(output):
    return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    print(f'Row: {inputs}')  # Debug: Check each row of data
    for layer in network:
        new_inputs = []
        for neuron in layer:
            print(f'Neuron weights before activation: {neuron["weights"]}')  # Debug: Check the neuron weights
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = np.array(new_inputs)
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = np.append(row[:-1], 1)  # add 1 for the bias
        if i != 0:
            inputs = np.append([neuron['output'] for neuron in network[i - 1]], 1)  # add 1 for the bias
        for neuron in network[i]:
            neuron['weights'][:-1] += l_rate * neuron['delta'] * inputs[:-1]
            neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

def calculate_accuracy(network, test_data):
    num_correct = 0
    for test in test_data:
        print(f'Test data: {test}')  # Debug: Check each test data item
        output = forward_propagate(network, test)
        if round(output[0]) == test[-1]: # Assumes binary classification
            num_correct += 1
    return num_correct / len(test_data) # Returns percentage of correct classifications

def crossover(parent1, parent2):
    child = list()
    for i in range(len(parent1)):
        child_layer = list()
        for j in range(len(parent1[i])):
            child_neuron = {'weights': (parent1[i][j]['weights'] + parent2[i][j]['weights']) / 2}
            child_layer.append(child_neuron)
        child.append(child_layer)
    return child

def mutate(network, mutation_rate):
    for layer in network:
        for neuron in layer:
            if random.random() < mutation_rate:
                neuron['weights'] = neuron['weights'] + np.random.uniform(-0.5, 0.5, len(neuron['weights']))

def genetic_algorithm(train_data, test_data, pop_size, n_generations, mutation_rate):
    population = [initialize_network(16, 10, 10, 1) for _ in range(pop_size)] # Example network structure
    for generation in range(n_generations):
        print(f'Generation {generation} population: {population}')  # Debug: Check the population each generation
        fitnesses = [calculate_accuracy(network, train_data) for network in population]
        next_generation = []
        for _ in range(pop_size):
            parents = random.choices(population, weights=fitnesses, k=2) # Selection
            child = crossover(parents[0], parents[1]) # Crossover
            mutate(child, mutation_rate) # Mutation
            next_generation.append(child)
        population = next_generation
        if (generation + 1) % 10 == 0: # Print out the maximum fitness every 10 generations
            print(f"Generation {generation + 1}, max fitness: {max(fitnesses)}")
    best_network = max(population, key=lambda network: calculate_accuracy(network, test_data)) # Evaluates the networks on the test data
    return best_network

def prepare_data(file):
    with open(file, "r") as f:
        data = f.readlines()

    inputs, outputs = [], []
    for line in data:
        split_line = line.split()
        inputs.append(np.array([int(char) for char in list(split_line[0])]))  # convert each character to an integer and make it numpy array
        outputs.append(np.array([int(split_line[1])]))

    train_size = int(0.8 * len(inputs))  # 80% of the data
    X_train, X_test = inputs[:train_size], inputs[train_size:]
    y_train, y_test = outputs[:train_size], outputs[train_size:]

    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = prepare_data("nn0.txt")

    # Convert training data to the format expected by our functions
    train_data = [[X, y] for X, y in zip(X_train, y_train)]
    test_data = [[X, y] for X, y in zip(X_test, y_test)]

    best_network = genetic_algorithm(train_data, test_data, pop_size=50, n_generations=100, mutation_rate=0.01)

    print(f"Best network accuracy on test data: {calculate_accuracy(best_network, test_data)}")

if __name__ == "__main__":
    main()

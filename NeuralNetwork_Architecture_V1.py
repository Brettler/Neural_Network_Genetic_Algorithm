import random
import numpy as np

class Agent:
    def __init__(self, layer_sizes, dropout_rate=0.2):
        self.layer_sizes = layer_sizes
        self.neural_network = NeuralNetwork(layer_sizes, dropout_rate)
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.2):
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]))
            self.biases.append(np.random.randn(layer_sizes[i + 1], 1))
        self.dropout_rate = dropout_rate
        self.n_features = layer_sizes[0]  # layer_sizes[0] is the input layer size

    def propagate(self, X, training=True):
        X = np.array(X).reshape(-1, self.n_features)  # Ensure X has the correct shape

        layer_input = X
        for i in range(len(self.weights) - 1):  # Loop over the layers
            print(f"layer_input shape: {layer_input.shape}")
            print(f"weights[i] shape: {self.weights[i].shape}")
            print(f"weights[i] slice shape: {self.weights[i][:len(layer_input[0])].shape}")

            hidden = self.relu(np.dot(layer_input, self.weights[i].T) + self.biases[i].T)

            if training:
                # Apply dropout to the layer
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=hidden.shape) / (1 - self.dropout_rate)
                hidden *= mask

            layer_input = hidden  # Output of current layer is input for next layer

        # Output layer (no dropout)
        output_layer = self.sigmoid(np.dot(layer_input, self.weights[-1][:len(layer_input[0])].T) + self.biases[-1].T)

        return output_layer

    def relu(self, x):
        return np.maximum(0, x)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def generate_agents(population_size, min_layers, max_layers, min_nodes, max_nodes):
    agents = []
    for _ in range(population_size):
        layer_sizes = generate_layer_sizes(min_layers, max_layers, min_nodes, max_nodes)
        agents.append(Agent(layer_sizes))
    return agents



def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    agents = agents[:int(0.2 * len(agents))]
    return agents


def unflatten(flattened, shapes):
    newarray = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        newarray.append(flattened[index : index + size].reshape(shape))
        index += size
    return newarray


def blend_crossover(alpha, parent1, parent2):
    # Create copies of the parents' genes
    genes1 = parent1.flatten()
    genes2 = parent2.flatten()

    # Initialize children genes with parent genes
    child1_genes = genes1.copy()
    child2_genes = genes2.copy()

    # Apply blend crossover to each gene
    for i in range(len(genes1)):
        # Calculate lower and upper bounds for the new genes
        lower = min(genes1[i], genes2[i]) - alpha * abs(genes1[i] - genes2[i])
        upper = max(genes1[i], genes2[i]) + alpha * abs(genes1[i] - genes2[i])

        # Generate the new genes by picking a random value between the lower and upper bounds
        child1_genes[i] = np.random.uniform(lower, upper)
        child2_genes[i] = np.random.uniform(lower, upper)

    # Reshape child genes to parent gene shapes
    child1_genes = child1_genes.reshape(parent1.shape)
    child2_genes = child2_genes.reshape(parent2.shape)

    return child1_genes, child2_genes

def crossover_weights_bias(agents, pop_size, min_layers, max_layers, min_nodes, max_nodes, alpha=0.5):
    offspring = []
    num_offspring = pop_size - len(agents)
    for _ in range(num_offspring // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)

        child1 = generate_agents(1, min_layers, max_layers, min_nodes, max_nodes)[0]
        child2 = generate_agents(1, min_layers, max_layers, min_nodes, max_nodes)[0]

        child1_weights = []
        child2_weights = []
        for w1, w2 in zip(parent1.neural_network.weights, parent2.neural_network.weights):
            child1_w, child2_w = blend_crossover(alpha, w1, w2)
            child1_weights.append(child1_w)
            child2_weights.append(child2_w)

        child1_biases = []
        child2_biases = []
        for b1, b2 in zip(parent1.neural_network.biases, parent2.neural_network.biases):
            child1_b, child2_b = blend_crossover(alpha, b1, b2)
            child1_biases.append(child1_b)
            child2_biases.append(child2_b)

        child1.neural_network.weights = child1_weights
        child2.neural_network.weights = child2_weights
        child1.neural_network.biases = child1_biases
        child2.neural_network.biases = child2_biases

        offspring.append(child1)
        offspring.append(child2)

    agents = agents[:pop_size - len(offspring)] + offspring

    return agents


def crossover(agents, pop_size):
    offspring = []
    num_offspring = pop_size - len(agents)
    for _ in range(num_offspring // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)

        child1_layer_sizes = parent1.layer_sizes[:]
        child2_layer_sizes = parent2.layer_sizes[:]

        # Perform crossover on layer_sizes
        for i in range(min(len(parent1.layer_sizes), len(parent2.layer_sizes))):
            if random.random() < 0.5:  # crossover probability
                child1_layer_sizes[i] = parent2.layer_sizes[i]
                child2_layer_sizes[i] = parent1.layer_sizes[i]

        child1 = Agent(child1_layer_sizes)
        child2 = Agent(child2_layer_sizes)
        offspring.append(child1)
        offspring.append(child2)

        offspring.append(child1)
        offspring.append(child2)

    agents = agents[:pop_size - len(offspring)] + offspring

    # Elitism: If there is a best agent, replace the worst performing agent

    # agents.sort(key=lambda agent: agent.fitness, reverse=True)
    # if agents[0].fitness > best_solution.fitness:
    #     agents[0] = best_agent

    return agents


# def mutation(agents):
#     mutation_rate = 0.2
#     for agent in agents:
#         # We start from index 1 and end at index -2 to avoid mutating input and output layer sizes
#         if len(agent.layer_sizes) > 2:
#             if random.uniform(0.0, 1.0) <= 0.2:
#                 # There are more than just input and output layers, so we can choose a hidden layer to mutate
#                 layer_to_mutate = random.randint(1, len(agent.layer_sizes) - 2)
#                 agent.layer_sizes[layer_to_mutate] = random.randint(min_nodes, max_nodes)
#         else:
#             # There are only input and output layers, so we can't mutate any layer sizes.
#             # You might want to add some other handling logic here if this situation shouldn't happen.
#             print("Warning: No hidden layers to mutate for this agent.")
#
#     return agents


def mutation_architecture(agents, min_nodes, max_nodes, min_layers, max_layers):
    input_features = 16
    for i, agent in enumerate(agents):
        if random.uniform(0.0, 1.0) <= 0.3:
            print(f"\nMutating agent {i}")
            # Decide whether to add, remove or change a layer
            mutation_type = random.choice(["add", "remove", "change"])
            print(f"Mutation type: {mutation_type}")

            if mutation_type == "add":
                # Add a new layer at a random position
                new_layer_size = random.randint(min_nodes, max_nodes)
                position = random.randint(1, len(agent.layer_sizes) - 1)  # -1 to exclude the output layer
                print(f"Adding layer at position {position} with size {new_layer_size}")
                agent.layer_sizes.insert(position, new_layer_size)

                # Also add new random weights and biases
                agent.neural_network.weights.insert(position,
                                                    np.random.randn(agent.layer_sizes[position], new_layer_size))
                agent.neural_network.biases.insert(position, np.random.randn(new_layer_size, 1))

                # Adjust weights and biases for the preceding layer
                agent.neural_network.weights[position - 1] = np.random.randn(new_layer_size,
                                                                             agent.layer_sizes[position - 1])
                agent.neural_network.biases[position - 1] = np.random.randn(new_layer_size, 1)

                # Adjust weights and biases for the next layer
                if position < len(agent.layer_sizes) - 1:  # Check if it's not the last layer
                    agent.neural_network.weights[position] = np.random.randn(agent.layer_sizes[position + 1],
                                                                             new_layer_size)
                    agent.neural_network.biases[position] = np.random.randn(agent.layer_sizes[position + 1], 1)

            if mutation_type == "remove" and len(agent.layer_sizes) > 3:  # Check if there are layers to remove (excluding input and output layers)
                # Remove a random layer
                position = random.randint(1, len(agent.layer_sizes) - 2)  # -2 to exclude the output layer
                print(f"Removing layer at position {position}")
                del agent.layer_sizes[position]

                # Also remove corresponding weights and biases
                del agent.neural_network.weights[position - 1]
                del agent.neural_network.biases[position - 1]

                # Adjust weights and biases for the preceding layer
                if position - 2 >= 0:  # Check if it's not the first layer
                    agent.neural_network.weights[position - 2] = np.random.randn(agent.layer_sizes[position - 1],
                                                                                 agent.layer_sizes[position - 2])
                    agent.neural_network.biases[position - 2] = np.random.randn(agent.layer_sizes[position - 1], 1)

            if mutation_type == "change":
                # Change the size of a random layer
                layer_to_mutate = random.randint(1, len(agent.layer_sizes) - 2)  # Exclude input and output layers

                # Ensure that first hidden layer has at least input_features nodes
                min_nodes_layer = input_features if layer_to_mutate == 1 else min_nodes

                new_layer_size = random.randint(min_nodes_layer, max_nodes)
                print(f"Changing layer {layer_to_mutate} size to {new_layer_size}")
                agent.layer_sizes[layer_to_mutate] = new_layer_size

                # Adjust weights and biases for the mutated layer
                agent.neural_network.weights[layer_to_mutate - 1] = np.random.randn(new_layer_size,
                    agent.layer_sizes[layer_to_mutate - 1])
                agent.neural_network.biases[layer_to_mutate - 1] = np.random.randn(new_layer_size, 1)

                # Adjust weights and biases for the succeeding layer
                if layer_to_mutate < len(agent.layer_sizes) - 1:  # Check if it's not the last layer
                    agent.neural_network.weights[layer_to_mutate] = np.random.randn(
                        agent.layer_sizes[layer_to_mutate + 1],
                        new_layer_size)
                    agent.neural_network.biases[layer_to_mutate] = np.random.randn(
                        agent.layer_sizes[layer_to_mutate + 1], 1)

    return agents



def mutation_weights_bias(agents):
    for agent in agents:
        if random.uniform(0.0, 1.0) <= 0.2:
            weights = agent.neural_network.weights
            biases = agent.neural_network.biases
            shapes = [a.shape for a in weights] + [b.shape for b in biases]
            flattened = np.concatenate([a.flatten() for a in weights] + [b.flatten() for b in biases])
            # random index will be used to select a random element for mutation.
            randint = random.randint(0, len(flattened) - 1)
            # The value at the randomly selected index in flattened is replaced
            # with a new random value generated using np.random.randn().
            flattened[randint] = np.random.randn()
            newarray = []
            indeweights = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
                indeweights += size
            agent.neural_network.weights = newarray[:len(weights)]
            agent.neural_network.biases = newarray[len(weights):]
    return agents

# This loss function is Mean Squared Error (MSE)
# def fitness(agents, X, y):
#     epsilon = 1e-7  # To prevent division by zero
#     for agent in agents:
#         yhat = agent.neural_network.propagate(X)
#         yhat = np.clip(yhat, epsilon, 1. - epsilon)  # Ensure yhat is within [epsilon, 1-epsilon]
#         log_loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
#         agent.fitness = log_loss
#     return agents


def fitness(agents, X, y, batch_size):
    epsilon = 1e-7  # To prevent division by zero
    num_samples = X.shape[0]

    for agent in agents:
        log_loss_list = []
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            yhat = agent.neural_network.propagate(X_batch)
            yhat = np.clip(yhat, epsilon, 1. - epsilon)  # Ensure yhat is within [epsilon, 1-epsilon]
            log_loss = -np.mean(y_batch * np.log(yhat) + (1 - y_batch) * np.log(1 - yhat))
            log_loss_list.append(log_loss)

        agent.fitness = np.mean(log_loss_list)  # Average log loss over all batches
    return agents


def calculate_accuracy(agent, X, y, isTraining = True):
    predictions = agent.neural_network.propagate(X, isTraining)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == y)
    return accuracy

def generate_layer_sizes(min_layers, max_layers, min_nodes, max_nodes):
    num_layers = np.random.randint(min_layers, max_layers + 1)
    layer_sizes = [np.random.randint(min_nodes, max_nodes + 1) for _ in range(num_layers)]
    layer_sizes[0] = 16  # Set the input layer to always have 16 nodes
    layer_sizes[-1] = 1  # Set the output layer to always have 1 node
    return layer_sizes

def execute(X_train, y_train, X_test, y_test, population_size, generations, min_layers, max_layers, min_nodes, max_nodes):
    batch_size = 124  # For example

    # Generate initial population
    agents = generate_agents(population_size, min_layers, max_layers, min_nodes, max_nodes)
    best_solution = agents[0]

    for i in range(generations):
        print('Generation', i, ':')
        agents = fitness(agents, X_train, y_train, batch_size)

        # Sort agents by fitness in ascending order
        #agents = sorted(agents, key=lambda agent: agent.fitness)

        agents = selection(agents)
        # Apply crossover and mutation
        agents = crossover(agents, population_size)
        agents = crossover_weights_bias(agents, population_size, min_layers, max_layers, min_nodes, max_nodes, alpha=0.5)

        agents = mutation_architecture(agents, min_nodes, max_nodes, min_layers, max_layers)
        agents = mutation_weights_bias(agents)
        agents = fitness(agents, X_train, y_train, batch_size)

        best_agent = min(agents, key=lambda agent: agent.fitness)
        if best_agent.fitness < best_solution.fitness:
            best_solution = best_agent

        train_loss = best_agent.fitness
        train_accuracy = calculate_accuracy(best_agent, X_train, y_train)
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        # Print the properties of the best agent
        print(f"Best agent properties: {len(best_agent.layer_sizes)} layers, Layer sizes: {best_agent.layer_sizes}")

    for agent in agents:
        isTrain = False
        train_loss = agent.fitness
        test_accuracy = calculate_accuracy(agent, X_test, y_test, isTrain)
        train_accuracy = calculate_accuracy(agent, X_train, y_train, isTrain)

        print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    isTrain = False
    train_loss = best_solution.fitness
    test_accuracy = calculate_accuracy(best_solution, X_test, y_test, isTrain)
    train_accuracy = calculate_accuracy(best_solution, X_train, y_train, isTrain)
    print("Best solution: ")
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    # After all generations are over, print properties of the best solution
    print("Best solution properties:")
    print(f"Number of layers: {len(best_solution.layer_sizes)}")
    print(f"Layer sizes: {best_solution.layer_sizes}")

    save_network(best_solution, "best_solution.txt")
    return best_solution


def prepare_data(file, test_ratio=0.25):
    with open(file, "r") as f:
        data = f.readlines()

    inputs, outputs = [], []
    for line in data:
        split_line = line.split()
        inputs.append([int(char) for char in split_line[0]])
        outputs.append([int(split_line[1])])

    X = np.array(inputs)
    y = np.array(outputs)

    # shuffle indices to make the split random
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # calculate the test set size
    test_set_size = int(X.shape[0] * test_ratio)

    X_test = X[indices[:test_set_size]]
    y_test = y[indices[:test_set_size]]

    X_train = X[indices[test_set_size:]]
    y_train = y[indices[test_set_size:]]

    return X_train, X_test, y_train, y_test


def save_network(agent, filename):
    with open(filename, "w") as f:
        # Saving hidden layer weights
        f.write(f"{agent.neural_network.weights[0].shape}\n")
        np.savetxt(f, agent.neural_network.weights[0])

        # Saving output layer weights
        f.write(f"{agent.neural_network.weights[1].shape}\n")
        np.savetxt(f, agent.neural_network.weights[1])

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("nn0.txt")
    input_dim = X_train.shape[1]
    print(f"input_dim is: {input_dim}")
    hidden1_dim = 15
    hidden2_dim = 7
    output_dim = y_train.shape[1]
    print(f"input_dim is: {output_dim}")
    population_size = 120
    generations = 120
    # Define the minimum and maximum number of layers and nodes per layer for the neural network.
    min_layers = 3
    max_layers = 5
    min_nodes = 4
    max_nodes = 20
    best_agent = execute(X_train, y_train, X_test, y_test, population_size, generations, min_layers, max_layers, min_nodes, max_nodes)
    print(f"Best Agent's fitness: {best_agent.fitness}")

import random
import numpy as np

class Agent:
    def __init__(self, network):
        self.neural_network = network
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, dropout_rate=0.2):
        self.weights = [
            np.random.randn(hidden1_dim, input_dim),
            np.random.randn(hidden2_dim, hidden1_dim),
            np.random.randn(output_dim, hidden2_dim),

        ]
        self.biases = [
            np.random.randn(hidden1_dim, 1),
            np.random.randn(hidden2_dim, 1),
            np.random.randn(output_dim, 1)
        ]
        self.dropout_rate = dropout_rate

    def propagate(self, X, training=True):
        X = np.array(X).reshape(-1, self.weights[0].shape[1])  # Ensure X has the correct shape

        hidden1 = self.relu(np.dot(X, self.weights[0].T) + self.biases[0].T)
        hidden2 = self.relu(np.dot(hidden1, self.weights[1].T) + self.biases[1].T)

        if training:
            # Apply dropout to hidden layers
            mask1 = np.random.binomial(1, 1 - self.dropout_rate, size=hidden1.shape) / (1 - self.dropout_rate)
            hidden1 *= mask1

            mask2 = np.random.binomial(1, 1 - self.dropout_rate, size=hidden2.shape) / (1 - self.dropout_rate)
            hidden2 *= mask2

        output_layer = self.sigmoid(np.dot(hidden2, self.weights[2].T) + self.biases[2].T)

        return output_layer

    def relu(self, x):
        return np.maximum(0, x)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def generate_agents(population, input_dim, hidden1_dim, hidden2_dim, output_dim):
    return [Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, output_dim)) for _ in range(population)]


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


def crossover(agents, pop_size, alpha=0.5, best_agent=None):
    offspring = []
    num_offspring = pop_size - len(agents)
    for _ in range(num_offspring // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)

        child1 = Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, output_dim))
        child2 = Agent(NeuralNetwork(input_dim, hidden1_dim, hidden2_dim, output_dim))

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

    # Elitism: If there is a best agent, replace the worst performing agent
    if best_agent is not None:
        agents.sort(key=lambda agent: agent.fitness, reverse=True)
        agents[0] = best_agent

    return agents


def mutation(agents):
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


def calculate_accuracy(agent, X, y, isTraining = True):
    predictions = agent.neural_network.propagate(X, isTraining)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == y)
    return accuracy

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

def execute(X_train, y_train, X_test, y_test, input_dim, hidden1_dim, hidden2_dim, output_dim, population_size, generations):
    agents = generate_agents(population_size, input_dim, hidden1_dim, hidden2_dim, output_dim)
    batch_size = 256
    best_solution = agents[0]

    for i in range(generations):
        print('Generation', i, ':')
        agents = fitness(agents, X_train, y_train, batch_size)

        # Sort agents by fitness in ascending order
        #agents = sorted(agents, key=lambda agent: agent.fitness)

        agents = selection(agents)
        # Apply crossover and mutation
        agents = crossover(agents, population_size, best_agent=best_solution)
        agents = mutation(agents)
        agents = fitness(agents, X_train, y_train, batch_size)

        best_agent = min(agents, key=lambda agent: agent.fitness)
        if best_agent.fitness < best_solution.fitness:
            best_solution = best_agent

        train_loss = best_agent.fitness
        train_accuracy = calculate_accuracy(best_agent, X_train, y_train)
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

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


    save_network(best_solution, "best_solution.txt")
    return best_solution


def prepare_data(file, test_ratio=0.2):
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
    X_train, X_test, y_train, y_test = prepare_data("nn1.txt")
    input_dim = X_train.shape[1]
    print(f"input_dim is: {input_dim}")
    hidden1_dim = 15
    hidden2_dim = 7
    output_dim = y_train.shape[1]
    print(f"input_dim is: {output_dim}")
    population_size = 100
    generations = 120
    best_agent = execute(X_train, y_train, X_test, y_test, input_dim, hidden1_dim, hidden2_dim, output_dim, population_size, generations)
    print(f"Best Agent's fitness: {best_agent.fitness}")

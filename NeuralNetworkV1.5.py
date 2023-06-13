import random
import numpy as np
from sklearn.model_selection import train_test_split

class Agent:
    def __init__(self, network):
        self.neural_network = network
        self.fitness = 0

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights = [np.random.randn(hidden_dim, input_dim), np.random.randn(output_dim, hidden_dim)]

    def propagate(self, X):
        X = np.array(X).reshape(-1, self.weights[0].shape[1])  # Ensure X has the correct shape
        hidden_layer = self.sigmoid(np.dot(X, self.weights[0].T))
        output_layer = self.sigmoid(np.dot(hidden_layer, self.weights[1].T))
        return output_layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def generate_agents(population, network):
    return [Agent(network) for _ in range(population)]

def fitness(agents, X, y):
    for agent in agents:
        yhat = agent.neural_network.propagate(X)
        cost = (yhat - y)**2
        agent.fitness = sum(cost)
    return agents

def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    #print('\n'.join(map(str, agents)))
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

def crossover(agents, network, pop_size):
    offspring = []
    for _ in range((pop_size - len(agents)) // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)
        child1 = Agent(network)
        child2 = Agent(network)

        shapes = [a.shape for a in parent1.neural_network.weights]

        genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
        genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])

        split = random.randint(0, len(genes1) - 1)
        child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
        child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())

        child1.neural_network.weights = unflatten(child1_genes, shapes)
        child2.neural_network.weights = unflatten(child2_genes, shapes)

        offspring.append(child1)
        offspring.append(child2)
    agents.extend(offspring)
    return agents

def mutation(agents):
    for agent in agents:
        if random.uniform(0.0, 1.0) <= 0.1:
            weights = agent.neural_network.weights
            shapes = [a.shape for a in weights]
            flattened = np.concatenate([a.flatten() for a in weights])
            randint = random.randint(0, len(flattened) - 1)
            flattened[randint] = np.random.randn()
            newarray = []
            indeweights = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[indeweights : indeweights + size].reshape(shape))
                indeweights += size
            agent.neural_network.weights = newarray
    return agents

def prepare_data(file):
    with open(file, "r") as f:
        data = f.readlines()

    inputs, outputs = [], []
    for line in data:
        split_line = line.split()
        inputs.append([int(char) for char in split_line[0]])
        outputs.append([int(split_line[1])])

    X = np.array(inputs)
    y = np.array(outputs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def save_network(agent, filename):
    with open(filename, "w") as f:
        # Saving hidden layer weights
        f.write(f"{agent.neural_network.weights[0].shape}\n")
        np.savetxt(f, agent.neural_network.weights[0])

        # Saving output layer weights
        f.write(f"{agent.neural_network.weights[1].shape}\n")
        np.savetxt(f, agent.neural_network.weights[1])

def calculate_accuracy(agent, X, y):
    predictions = agent.neural_network.propagate(X)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == y)
    return accuracy

def execute(X_train, y_train, X_test, y_test, input_dim, hidden_dim, output_dim, population_size, generations, threshold):
    network = NeuralNetwork(input_dim, hidden_dim, output_dim)
    agents = generate_agents(population_size, network)

    for i in range(generations):
        print('Generation', i, ':')
        agents = fitness(agents, X_train, y_train)
        agents = selection(agents)
        agents = crossover(agents, network, population_size)
        agents = mutation(agents)
        agents = fitness(agents, X_train, y_train)

        best_agent = min(agents, key=lambda agent: agent.fitness)
        train_loss = best_agent.fitness
        train_accuracy = calculate_accuracy(best_agent, X_train, y_train)
        test_accuracy = calculate_accuracy(best_agent, X_test, y_test)
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        if train_loss < threshold:
            print('Threshold met at generation', i, '!')
            break

    return best_agent

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("nn0.txt")
    input_dim = X_train.shape[1]
    hidden_dim = 10
    output_dim = y_train.shape[1]
    population_size = 100
    generations = 150
    threshold = 0.01
    best_agent = execute(X_train, y_train, X_test, y_test, input_dim, hidden_dim, output_dim, population_size, generations, threshold)
    save_network(best_agent, "wnet.txt")
    print(f"Best Agent's fitness: {best_agent.fitness}")

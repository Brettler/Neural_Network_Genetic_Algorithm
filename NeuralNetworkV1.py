import json

import numpy as np
from sklearn.model_selection import train_test_split


class Agent:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initializing random weights and biases for hidden and output layer
        self.weights_hidden = np.random.randn(hidden_dim, input_dim)
        self.bias_hidden = np.random.randn(hidden_dim, 1)

        self.weights_output = np.random.randn(output_dim, hidden_dim)
        self.bias_output = np.random.randn(output_dim, 1)
        # Add train and test fitness
        self.train_fitness = 0
        self.test_fitness = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, X):
        X = np.array(X).reshape(1, -1)  # ensure that X is a 2D array
        self.hidden_layer = self.sigmoid(np.dot(self.weights_hidden, X.T) + self.bias_hidden)
        self.output_layer = self.sigmoid(np.dot(self.weights_output, self.hidden_layer) + self.bias_output)
        return self.output_layer

    def copy(self):
        new_agent = Agent(self.weights_hidden.shape[1], self.weights_hidden.shape[0], self.weights_output.shape[0])
        new_agent.weights_hidden = self.weights_hidden.copy()
        new_agent.bias_hidden = self.bias_hidden.copy()
        new_agent.weights_output = self.weights_output.copy()
        new_agent.bias_output = self.bias_output.copy()
        return new_agent

    def mutate(self, mutation_rate):
        # Mutate weights and biases with a certain probability (mutation rate)
        for w in [self.weights_hidden, self.weights_output]:
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    if np.random.random() < mutation_rate:
                        #print('Mutating...')
                        w[i][j] += np.random.randn()

        for b in [self.bias_hidden, self.bias_output]:
            for i in range(b.shape[0]):
                if np.random.random() < mutation_rate:
                    b[i] += np.random.randn()


class GeneticAlgorithm:
    def __init__(self, population_size, input_dim, hidden_dim, output_dim, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.agents = [Agent(input_dim, hidden_dim, output_dim) for _ in range(population_size)]

    def selection(self, fitnesses):
        print(f'Fitnesses before selection: {fitnesses}')
        # Select two parents based on their fitness probabilities
        fitnesses = fitnesses / (np.sum(fitnesses) + 1e-7)  # prevent division by zero , normalize fitnesses to probabilities
        fitnesses = fitnesses / np.sum(fitnesses)  # ensure sum is 1

        idx = np.arange(self.population_size)
        parent_idx1 = np.random.choice(idx, p=fitnesses)
        parent_idx2 = np.random.choice(idx, p=fitnesses)
        while parent_idx2 == parent_idx1:  # ensure we have two different parents
            parent_idx2 = np.random.choice(idx, p=fitnesses)
        return self.agents[parent_idx1], self.agents[parent_idx2]

    def crossover(self, agent1, agent2):
        # Single-point crossover
        new_agent = agent1.copy()
        crossover_idx = np.random.randint(low=0, high=new_agent.weights_hidden.shape[1])

        # Crossing over weights and biases for the hidden layer
        new_agent.weights_hidden[:, crossover_idx:] = agent2.weights_hidden[:, crossover_idx:]
        new_agent.bias_hidden = agent2.bias_hidden if np.random.random() < 0.5 else agent1.bias_hidden

        # Crossing over weights and biases for the output layer
        new_agent.weights_output[:, crossover_idx:] = agent2.weights_output[:, crossover_idx:]
        new_agent.bias_output = agent2.bias_output if np.random.random() < 0.5 else agent1.bias_output

        return new_agent

    def evolve(self, fitnesses):
        # Create a new population
        new_population = []
        for _ in range(self.population_size):
            # Selection
            parent1, parent2 = self.selection(fitnesses)
            # Crossover
            child = self.crossover(parent1, parent2)
            # Mutation
            child.mutate(self.mutation_rate)
            print(f"Child in new population: {child}")
            new_population.append(child)
        self.agents = new_population  # replace the old population with the new one


    def execute(self, X_train, y_train, X_test, y_test):
        # Calculate fitness
        for agent in self.agents:
            agent.train_fitness = self.calculate_fitness(agent, X_train, y_train)
            agent.test_fitness = self.calculate_fitness(agent, X_test, y_test)

        # Sorting agents based on fitness
        self.agents.sort(key=lambda agent: agent.train_fitness, reverse=True)
        print([agent.train_fitness for agent in self.agents])
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")
            print([agent.train_fitness for agent in self.agents])

            best_agent = self.agents[0]
            print(f"Best Agent's train accuracy: {best_agent.train_fitness}, test accuracy: {best_agent.test_fitness}")
            if best_agent.train_fitness >= 0.99:  # Modify the condition based on your desired accuracy threshold
                print(f"Desired accuracy reached at Generation {generation + 1}")
                return best_agent
            self.evolve([agent.train_fitness for agent in self.agents])
        return best_agent

    def calculate_fitness(self, agent, X, y):
        num_correct_predictions = 0
        for i in range(len(X)):
            prediction = agent.feedforward(X[i])
            predicted_label = np.round(prediction)
            if predicted_label == y[i]:
                num_correct_predictions += 1
        accuracy = num_correct_predictions / len(X)
        return accuracy

    # def calculate_fitness(self, agent, X, y):
    #     num_correct_predictions = 0
    #     for i in range(len(X)):
    #         prediction = agent.feedforward(X[i])
    #         predicted_label = np.round(prediction)[0]  # Extract the single predicted label
    #         if predicted_label == y[i][0]:  # Compare individual elements of predicted_label and y[i]
    #             num_correct_predictions += 1
    #     accuracy = num_correct_predictions / len(X)
    #     return accuracy

def prepare_data(file):
    # Open the file and read its contents
    with open(file, "r") as f:
        data = f.readlines()

    # Initialize empty lists to store inputs and outputs
    inputs, outputs = [], []

    # Process each line in the file
    for line in data:
        # Split the line into input and output components
        split_line = line.split()

        # Convert the input string into a list of integers
        inputs.append([int(char) for char in split_line[0]])

        # Convert the output string into a single integer
        outputs.append([int(split_line[1])])

    # Convert the input and output lists into numpy arrays
    X = np.array(inputs)
    y = np.array(outputs)

    # Split the data into training and testing sets
    # using a test size of 20% and a fixed random seed of 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Return the train-test split data
    return X_train, X_test, y_train, y_test
def save_network(agent, filename):
    with open(filename, "w") as f:
        # Saving hidden layer weights
        f.write(f"{agent.weights_hidden.shape}\n")
        np.savetxt(f, agent.weights_hidden)

        # Saving hidden layer biases
        f.write(f"{agent.bias_hidden.shape}\n")
        np.savetxt(f, agent.bias_hidden)

        # Saving output layer weights
        f.write(f"{agent.weights_output.shape}\n")
        np.savetxt(f, agent.weights_output)

        # Saving output layer biases
        f.write(f"{agent.bias_output.shape}\n")
        np.savetxt(f, agent.bias_output)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data("nn0.txt")
    input_dim = 16
    hidden_dim = 10  # specify hidden dimension as per your requirements
    output_dim = 1
    ga = GeneticAlgorithm(population_size=50, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, mutation_rate=0.4, generations=100)
    best_agent = ga.execute(X_train, y_train, X_test, y_test)
    save_network(best_agent, "wnet.txt")
    print(f"Best Agent's train fitness: {best_agent.train_fitness}, test fitness: {best_agent.test_fitness}")


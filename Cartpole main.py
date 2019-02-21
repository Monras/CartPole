# By Måns Rasmussen and Gustaf Holte
"""Simulates an inverted pendulum and balance it using Neural Network coding"""

"""
Att göra/testa: 
 - Ändra action från 1,0 till andra diskreta värden på force
 - Hitta bättre träningsdata

"""

# imports packages
import csv
from gym_kod import CartPoleEnv as gym
import random as rnd
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import matplotlib.pyplot as plt

# Variables
LR = 1e-3  # Learning Rate
#global N_runs, N_steps_per_run, score_limit
N_runs = 100  # Number of different runs of cartpole simulations
N_steps_per_run = 50  # Maximum number of steps per run for each simulation
score_limit = 5  # minimum score limit
keep_prob = 0.8

def save_to_csvfile(simulations, steps_run, training_data):
    """Saves data to an csv file"""
    with open('training_data_test.csv', mode='w') as training_file:
        print(training_data)
        filewriter = csv.writer(training_file, delimiter=':', escapechar='"', quoting=csv.QUOTE_NONE)
        filewriter.writerow(['x x_dot theta theta_dot polelength polemass ', ' Action'])
        sim = 1
        for sim in range(simulations):
            print("sim: ", sim)
            filewriter.writerow(['Simulation ', sim])
            for steps in steps_run:
                i = 0
                for i in range(int(steps)-1):
                    print("step: ", i)
                    print("trainig_data: ", training_data[0][i])
                    print("trainig_data: ", training_data[1][i])
                    filewriter.writerow([training_data[0][i], training_data[1][i]])

def Neural_Network(input_size):
    """Creates a Neural network, returns a neural network"""
    print("input size: ", input_size)
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 16, activation='sigmoid')  # fully_connected creates one neural network layer
                                                            # Activation specifies which activation function we want to use
    network = dropout(network, keep_prob)

    network = fully_connected(network, 64, activation='sigmoid')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 128, activation='sigmoid')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 64, activation='sigmoid')
    network = dropout(network, keep_prob)

    #network = fully_connected(network, 1, activation='sigmoid')
    #network = dropout(network, keep_prob)

    network = fully_connected(network, 30, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                         name='targets')  # Creates a regression, optimizes the network using back-propagation
    NN = tflearn.DNN(network, tensorboard_dir='log')  # Creates the neural network

    return NN

def create_training_data():
    """Creates and saves training data that gets score above score_limit """
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    accepted_scores = []  # saves the accepted scores
    scores_arr = []

    for run in range(N_runs):
        if run % 10 == 0:
            print("run nr: ", run)
        env.reset()
        score = 0
        runs_memory = []  # saves the runs
        #prev_observation = []

        for _ in range(N_steps_per_run):
            action = rnd.randrange(start=-150, stop=150, step=10)  # creates a random number: between 0 and 150
            observation, reward, done, info = env.step(action)  # updates the observation with given action
            #env.render()  # updates the animation
            """saves the next observation and action"""
            runs_memory.append([observation, action])  # saves the action that gave that observation
            score += reward  # updates the score as long as done is not False
            if done:
                break
            if score >= score_limit:
                accepted_scores.append(score)
                for data in runs_memory:
                    """Checks if the action is 1 (force is positive) or 0 (force is negative, pushes to the left)"""
                    training_data.append([data[0], data[1]])

            #env.render()  # updates the animation
            #env.reset()

            scores.append(score)  # saves the score for that step
        scores_arr.append(score)

        # Saves the training data to a numpy file
        training_data_save = np.array(training_data)
        np.save('training_data_test.npy', N_runs, score_limit, training_data_save)  # saves the training data

    # some stats here, to further illustrate the neural network magic!
    #print('Average score:', mean(scores))
    #print('Median scores:', median(scores))

    plt.figure()
    plt.hist(scores_arr, bins=100)
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    plt.show()

    #save_to_csvfile(len(accepted_scores), accepted_scores, training_data)

    return training_data

def train_model(training_data=None, model=False):
    """Trains the neural network model with training_data"""
    if training_data is None:
        print("Loading training data")
        training_data = np.load("training_data.npy")
    print("training data length:", len(training_data[0][0]))
    print("training data:", training_data[0][0])
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    print("X: ", X)
    print("y: ", y)
    if not model:  # Creates a new model if no model is entered as attribute
        model = Neural_Network(input_size=len(X[0]))
    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

def random_simulation():
    """Runs N_runs simulations with a random action"""
    env = gym()  # creates the enviroment
    env.reset()
    # training_data = create_training_data()
    """Runs the simulations and the main code"""
    for i_episode in range(N_runs):
        """Runs the cartpole simulations N_runs times """
        env.reset()  # resets the pendulum to the initial position (with random initializations)
        for t in range(N_steps_per_run):
            """Updates each step for this cartpole run"""
            env.render()  # Runs the animation, updates for each step (turn off for faster computation!)
            action = env.action_space.sample()  # Creates a sample action (random!), i.e. left or right to the cart
            observation, reward, done, info = env.step(action)  # executes the envirement with the given action,/
            # returns the observation of that action on the enviroment
            # observation is the state variables of the cartpole
            if done:
                """If done is True, terminates the simulation. Either the goal is reached or it has failed."""
                print("Episode finished after {} timesteps".format(t + 1))
                break

def NN_simulation(env, NN):
    """Runs N_runs simulations using the trained Neural Network"""
    scores = []
    choices = []
    training_data = []
    for each_game in range(10):
        score = 0
        runs_memory = []
        prev_obs = []
        for _ in range(N_steps_per_run):
            env.render()
            if len(prev_obs) == 0:
                action = rnd.randrange(start=0, stop=150, step=10)
            else:
                action = np.argmax(NN.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            runs_memory.append([new_observation, action])
            score += reward
            if done: break
            for data in runs_memory:
                """Checks if the action is 1 (force is positive) or 0 (force is negative, pushes to the left)"""
                training_data.append([data[0], data[1]])
        env.reset()
        scores.append(score)

    print('Average Score:', sum(scores) / len(scores))
    print('actions 1 (to the right): {}%   actions 0 (to the left): {}%'.format((round(choices.count(1) / len(choices)*100)), round(choices.count(0) / len(choices)*100)))
    return training_data

if __name__ == "__main__":
    # random_simulation()
    env = gym()
    training_data = None
    NN = False
    training_data = create_training_data()
    #for row in training_data:
        #print("Action: ", row[1], "\n")
        #print("Observation : ", row[0])

    #training_data = np.load("training_data.npy")
    for _ in range(0, 1):
        env.reset()
        NN = train_model(training_data, NN)
        #np.save("trained_model.npy", NN)
        #model = tflearn.DNN(NN, tensorboard_verbose=3)
        training_data = NN_simulation(env, NN)


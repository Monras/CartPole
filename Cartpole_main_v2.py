# By Måns Rasmussen and Gustaf Holte

"""Simulates an inverted pendulum and balance it using Neural Network coding

    version 2:
        - Can load and save models and training data
        - Can train itself itteratevly with theta_limit getting narrower for each time as well as having different \
          score limits
        - Have 5 discrete force steps, -200 -100 0 100 200
        -
    To be fixed:
        - Fix generational improvements!
        - Save histograms
        - Optimera neural network
        - remove any hard coding


    """

# imports packages
import csv
from gym_kod_v2 import CartPoleEnv as gym
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
N_runs = 10000  # Number of different runs of cartpole simulations
N_steps_per_run_1 = 150  # Maximum number of steps per run for each simulation
N_steps_per_run_2 = 250
score_limit_1 = 50  # minimum score limit
score_limit_2 = 100
keep_prob = 0.8
epoch = 1
lower = -200
upper = 300
step = 100

def Neural_Network(input_size, output_size):
    """Creates a Neural network, returns a neural network"""

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 16, activation='sigmoid')  # fully_connected creates one neural network layer
                                                            # Activation specifies which activation function we want to use
    network = dropout(network, keep_prob)

    network = fully_connected(network, 64, activation='sigmoid', name="fc_layer_1")
    network = dropout(network, keep_prob)

    network = fully_connected(network, 128, activation='sigmoid', name="fc_layer_2")
    network = dropout(network, keep_prob)

    network = fully_connected(network, 64, activation='sigmoid', name="fc_layer_3")
    network = dropout(network, keep_prob)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                         name='targets')  # Creates a regression, optimizes the network using back-propagation
    model = tflearn.DNN(network, tensorboard_dir='log')  # Creates the neural network

    return model

def create_training_data(env, theta_limit, file_name=''):
    """Creates and saves training data that gets score above score_limit """
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    accepted_scores = []  # saves the accepted scores
    scores_arr = []

    for run in range(N_runs):
        if run % 1000 == 0:
            print("run nr: ", run)
        env.reset(theta_limit)
        score = 0
        runs_memory = []  # saves the runs
        prev_observation = []
        output = []
        """saves the next observation and action"""
        for _ in range(N_steps_per_run_1):
            action = rnd.randrange(start=lower, stop=upper, step=step)  # creates a random number: between -200 and 200, steps 100
            observation, reward, done, info = env.step(action)  # updates the observation with given action
            #env.render()  # updates the animation
            if len(prev_observation) > 0:
                runs_memory.append([prev_observation, action])  # saves the action that gave that observation
            prev_observation = observation
            score += reward  # updates the score as long as done is not False
            if done:
                break
            if score >= score_limit_1:
                accepted_scores.append(score)
                for data in runs_memory:
                    """Checks if the action is 1 (force is positive) or 0 (force is negative, pushes to the left)"""
                    if data[1] == -200:
                        output = np.array([1, 0, 0, 0, 0])
                    if data[1] == -100:
                        output = np.array([0, 1, 0, 0, 0])
                    if data[1] == 0:
                        output = np.array([0, 0, 1, 0, 0])
                    if data[1] == 100:
                        output = np.array([0, 0, 0, 1, 0])
                    if data[1] == 200:
                        output = np.array([0, 0, 0, 0, 1])

                    training_data.append([data[0], output])

            scores.append(score)  # saves the score for that step
        scores_arr.append(score)

    # Saves the training data to a numpy file
    training_data_save = np.array(training_data)
    np.save(file_name, training_data_save)  # saves the training data

    # some stats here, to further illustrate the neural network magic!
    #print('Average score:', mean(scores))
    #print('Median scores:', median(scores))

    plt.figure()
    plt.hist(scores_arr, bins=100)
    plt.title("Random moves simulation")
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    plt.show()


    return training_data

def train_model(training_data=None, model=None, new_model = False, save_model=False, file_name=''):
    """Trains the neural network model with training_data"""

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)  # Observation variables! From training data
    y = [i[1] for i in training_data]  # All forces

    if new_model is True:  # Creates a new model if no model is entered as attribute
        model = Neural_Network(input_size=len(X[0]), output_size=len(y[0]))
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True,
                  run_id='openai_learning')
        if save_model is True:
            # Save a model
            model.save(file_name)
        return model
    elif model is not None:
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True, run_id='openai_learning')
        if save_model is True:
            # Save a model
            model.save(file_name)
        return model
    else:
        raise ValueError("Something went wrong when training the model")

def load_model(file_name, input_size, output_size):
    """Loads and returns a NN model"""
    model = Neural_Network(input_size=input_size, output_size=output_size)
    model.load(file_name)
    return model

def random_simulation():
    """Runs N_runs simulations with a random action"""
    env = gym()  # creates the enviroment
    env.reset(theta_limit_1)
    # training_data = create_training_data()
    """Runs the simulations and the main code"""
    for i_episode in range(N_runs):
        """Runs the cartpole simulations N_runs times """
        env.reset(theta_limit_1)  # resets the pendulum to the initial position (with random initializations)
        for t in range(N_steps_per_run_1):
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

def NN_simulation(env, NN, theta_lim, save_data=False, file_name=''):
    """Runs N_runs simulations using the trained Neural Network,
       returns the new training data"""
    scores = []
    choices = []
    training_data = []
    scores_arr = []
    accepted_scores = []

    for run in range(N_runs):
        if run % 1000 == 0:
            print("run nr: ", run)
        env.reset(theta_lim)
        score = 0
        runs_memory = []
        prev_obs = []
        output = []
        for _ in range(N_steps_per_run_2):
            #env.render()
            if len(prev_obs) == 0:
                action = rnd.randrange(start=lower, stop=upper, step=step)
            else:
                action = np.argmax(NN.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])  # argmax returns the index of the output neuron with the largest weight
                action = np.arange(-200, 300, 100)[action]  # Translates the output to force/action applied to the cartpole

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            runs_memory.append([new_observation, action])
            score += reward
            if done:
                break
            if score >= score_limit_2:
                accepted_scores.append(score)
                for data in runs_memory:
                    """Translates the force/ action to an normalized array"""
                    if data[1] == -200:
                        output = np.array([1, 0, 0, 0, 0])
                    if data[1] == -100:
                        output = np.array([0, 1, 0, 0, 0])
                    if data[1] == 0:
                        output = np.array([0, 0, 1, 0, 0])
                    if data[1] == 100:
                        output = np.array([0, 0, 0, 1, 0])
                    if data[1] == 200:
                        output = np.array([0, 0, 0, 0, 1])

                    training_data.append([data[0], output])


            scores.append(score)  # saves the score for that step
        scores_arr.append(score)

    # Saves the training data to a numpy file
    if save_data:
        training_data_save = np.array(training_data)
        np.save(file_name, training_data_save)  # saves the training data

    # some stats here, to further illustrate the neural network magic!
    #print('Average score:', mean(scores))
    #print('Median scores:', median(scores))

    plt.figure()
    plt.hist(scores_arr, bins=100)
    plt.title("Learned NN simulation")
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    plt.show()
    print('Average Score:', sum(scores) / len(scores))
    print('actions 1 (to the right): {}%   actions 0 (to the left): {}%'.format((round(np.sum(np.array(choices) >= 0, axis=0) / len(choices)*100)), round(np.sum(np.array(choices) < 0, axis=0) / len(choices)*100)))

    return training_data

if __name__ == "__main__":
    # random_simulation()
    env = gym()
    theta_limit_1 = 88
    theta_limit_2 = 40
    theta_limit_3 = 20
    input_size = 6
    output_size = 5

    # Create new training_data (randomly!) and train a new model of that data
    #training_data_1 = create_training_data(env, theta_limit_1, file_name='training_data_random.npy')
    #model_1 = train_model(training_data_1, load_model=False, save_model=True, file_name='cartpole_model_1')

    # Load training_data
    #training_data_1 = np.load('training_data_1.npy')

    # Load model
    model_1 = load_model('cartpole_NN_model_1.tflearn', input_size, output_size)
    #model_2 = load_model('cartpole_model_2.tflearn', input_size, output_size)


    # Load model and training data, train new model of that
    #training_data_2 = NN_simulation(env, model_1, theta_limit_2,save_data=True, file_name='training_data_2.npy')
    #model_2 = train_model(training_data_2, model_1, save_model=True, file_name='cartpole_model_2.tflearn')

    training_data_3 = NN_simulation(env, model_1, theta_limit_2, save_data=False, file_name='training_data_3.npy')





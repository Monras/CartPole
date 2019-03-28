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
from gym_kod_v3 import CartPoleEnv as gym
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

def create_training_data(env, theta_limit, save=False, animation=False, N_runs=None, N_steps_per_run=None, score_limit=None, file_name=''):
    """Creates and saves training data that gets score above score_limit """
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    accepted_scores = []  # saves the accepted scores
    scores_arr = []
    div = N_runs / 10
    for run in range(N_runs):
        if run % div == 0:
            print("run nr: ", run)
        env.reset(theta_limit)
        score = 0
        runs_memory = []  # saves the runs
        prev_observation = []
        output = []
        """saves the next observation and action"""
        for _ in range(N_steps_per_run):
            action = rnd.randrange(start=lower, stop=upper, step=step)  # creates a random number: between -200 and 200, steps 100
            observation, reward, done, info = env.step(action)  # updates the observation with given action
            if animation:
                env.render()  # updates the animation
            if len(prev_observation) > 0:
                runs_memory.append([prev_observation, action])  # saves the action that gave that observation
            prev_observation = observation
            score += reward  # updates the score as long as done is not False
            if done:
                break
            if score >= score_limit:
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

    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.hist(scores_arr, bins=100)
    plt.title("Randomly created training data")
    hist_text = '\n'.join((' Score limit: ' + str(score_limit), " Theta limit: " + str(theta_limit),
                           "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                           "Mean scores: " + str(round(mean(scores),1)), "Median: " + str(round(median(scores),1))))
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    plt.text(0.9, 0.9, hist_text, transform=ax.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=props)
    hist_name = file_name[0:-4] + "_histogram"
    if save:
        # Saves the training data to a numpy file, if save is True!
        file_data = np.array(training_data)
        np.save(file_name, file_data)  # saves the training data
        plt.savefig(hist_name)  # saves the histogram
        print("Data saved as: ", file_name)
        print("Histogram saved as: ", file_name + "_histogram")
    plt.show()
    return training_data

def train_model(training_data=None, model=None, new_model=False, save=False, file_name=''):
    """Trains the neural network model with training_data"""
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)  # Observation variables! From training data
    y = [i[1] for i in training_data]  # All forces

    if new_model is True:  # Creates a new model if no model is entered as attribute
        model = Neural_Network(input_size=len(X[0]), output_size=len(y[0]))
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True,
                  run_id='openai_learning')
        if save is True:
            # Save a model
            model.save(file_name)
        return model
    elif model is not None:
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True, run_id='openai_learning')
        if save is True:
            # Save a model
            model.save(file_name)
            print("Model saved as: ", file_name)
        return model
    else:
        raise ValueError("Something went wrong when training the model")

def load_model(file_name, input_size, output_size):
    """Loads and returns a NN model"""
    print("Loading file: ", file_name)
    model = Neural_Network(input_size=input_size, output_size=output_size)
    model.load(file_name)
    return model

def random_simulation(N_runs):
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

def NN_simulation(env, NN, theta_lim, score_limit, N_runs, N_steps_per_run, save=False, animation=False, file_name=''):
    """Runs N_runs simulations using the trained Neural Network,
       returns the new training data"""
    scores = []
    choices = []
    training_data = []
    scores_arr = []
    accepted_scores = []
    div = N_runs/10
    for run in range(N_runs):
        if run % div == 0:
            print("run nr: ", run)
        env.reset(theta_lim)
        score = 0
        runs_memory = []
        prev_obs = []
        output = []
        for _ in range(N_steps_per_run):
            if animation:
                env.render()
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
            if score >= score_limit:
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

    fig, ax = plt.subplots()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.hist(scores_arr, bins=100)
    plt.title("NN made training data")
    hist_text = '\n'.join((' Score limit: ' + str(score_limit), " Theta limit: " + str(theta_lim),
                           "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                           "Mean scores: " + str(round(mean(scores), 1)), "Median: " + str(round(median(scores), 1))))
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    plt.text(0.9, 0.9, hist_text, transform=ax.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=props)
    hist_name = file_name[0:-4] + "_histogram"

    # Saves the training data to a numpy file
    if save:
        training_data_save = np.array(training_data)
        np.save(file_name, training_data_save)  # saves the training data
        plt.savefig(hist_name)  # saves the histogram
        print("Histogram saved as: ", file_name + "_histogram")
        print("Trained data saved as: ", file_name)

    plt.show()
    print('Average Score:', sum(scores) / len(scores))
    print('actions 1 (to the right): {}%   actions 0 (to the left): {}%'.format((round(np.sum(np.array(choices) >= 0, axis=0) / len(choices)*100)), round(np.sum(np.array(choices) < 0, axis=0) / len(choices)*100)))

    return training_data

if __name__ == "__main__":
    """Runs the main code"""
    # random_simulation()
    env = gym()
    theta_limit_1 = 90
    theta_limit_2 = 45
    theta_limit_3 = 20
    N_runs_1 = 10000  # Number of runs per simulation
    N_runs_2 = 20000
    N_runs_3 = 10000
    N_steps_per_run_1 = 100  # Maximum number of steps per run for each simulation
    N_steps_per_run_2 = 200
    N_steps_per_run_3 = 300
    score_limit_1 = 30  # minimum score limit
    score_limit_2 = 15
    score_limit_3 = 5
    score_limit_4 = 50
    input_size = 6
    output_size = 5


    """Load old models (make sure the file is in the same folder as the code!"""
    # model_1 = load_model('cartpole_model_v2_4_1.tflearn', input_size, output_size)
    # model_2 = load_model('cartpole_model_v2_4_2.tflearn', input_size, output_size)
    model_1 = load_model('cartpole_model_v2_4_2_1.tflearn', input_size, output_size)

    """Load old training_data_#"""
    # training_data_1 = np.load('training_data_v2_4_1.npy')
    # training_data_2 = np.load('training_data_v2_4_2.npy')
    # training_data_3 = np.load('training_data_v2_3_3.npy')
    # training_data_trained = np.load('training_data_v2_2_2_trained.npy')

    """Create new training_data (randomly!)"""
    """
    training_data_1 = create_training_data(env, theta_limit_1, score_limit=score_limit_1, save=True, file_name='training_data_v2_4_1.npy',
                                           N_runs=N_runs_1, N_steps_per_run=N_steps_per_run_1)

    training_data_2 = create_training_data(env, theta_limit_2, score_limit=score_limit_2, save=True, file_name='training_data_v2_4_2.npy',
                                           N_runs=N_runs_2, N_steps_per_run=N_steps_per_run_1)
                                           
    # training_data_3 = create_training_data(env, theta_limit_3, score_limit=score_limit_3, save=True, file_name='training_data_v2_4_3.npy',
                                           # N_runs=N_runs_3, N_steps_per_run=N_steps_per_run_3)

    print("training data 1 size: ", len(training_data_1))
    print("training data 2 size: ", len(training_data_2))
    # print("training data 3 size: ", len(training_data_3))
    """


    """Creates or trains new models based on the training data"""
    # model_1 = train_model(training_data_1, model=False, new_model=True, save=True, file_name='cartpole_model_v2_4_1.tflearn')
    # model_2 = train_model(training_data_2, model=model_1, new_model=False, save=True, file_name='cartpole_model_v2_4_2_1.tflearn')

    """Run new simulation with existing model and saves the new training data"""

    training_data_1_trained = NN_simulation(env, model_1, theta_limit_2, score_limit_4, N_runs_1, N_steps_per_run_1,
                                            save=False, animation=True, file_name='training_data_v2_4_1_trained.npy')
    """
    training_data_2_trained = NN_simulation(env, model_2, theta_limit_2, score_limit_2, N_runs_2, N_steps_per_run_2,
                                            save=False, animation=True, file_name='training_data_v2_4_2_trained.npy')
    """

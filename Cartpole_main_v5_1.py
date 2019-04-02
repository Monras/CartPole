# By Måns Rasmussen and Gustaf Holte

"""Simulates an inverted pendulum and balance it using Neural Network coding

    version 5:
        - Nya histogram med bra axlar
        - Mer allmänt för storleken på kraften
        - 
    To be fixed:
        - Optimera neural network
        - Hur många lager ska vi ha?
        - Vilken vinkel ska vi ha för simulationen?


    """

# imports packages
from gym_kod_v3_1 import CartPoleEnv as gym
import random as rnd
import numpy as np
import h5py
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
import matplotlib.pyplot as plt


# Variables
LR = 1e-3  # Learning Rate
keep_prob = 0.8
epoch = 1
step = 75  # tweak variable to find the right range of the force
lower = step
upper = step*3 + 1


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
    scores_no_zero = []
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
            action = rnd.randrange(start=lower, stop=upper, step=step)  # creates a random number in the given range
            if rnd.random() < 0.5:
                action = action
            else:
                action = action*-1
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
                    if data[1] == -step*3:
                        output = np.array([1, 0, 0, 0, 0, 0])
                    if data[1] == -step*2:
                        output = np.array([0, 1, 0, 0, 0, 0])
                    if data[1] == -step:
                        output = np.array([0, 0, 1, 0, 0, 0])
                    if data[1] == step:
                        output = np.array([0, 0, 0, 1, 0, 0])
                    if data[1] == step*2:
                        output = np.array([0, 0, 0, 0, 1, 0])
                    if data[1] == step*3:
                        output = np.array([0, 0, 0, 0, 0, 1])
                    training_data.append([data[0], output])
            if score != 0:
                scores_no_zero.append(score)
            scores.append(score)  # saves the score for that step
        scores_arr.append(score)
    N_bins = int(max(scores))
    fig, ax = plt.subplots(tight_layout=True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.hist(scores_arr, bins=N_bins)
    font = {'family': 'normal',
            'size': 12}
    plt.rc('font', **font)
    plt.title("Randomly created training data", fontsize=20)

    hist_text = '\n'.join((' Score limit: ' + str(score_limit), " Theta limit: " + str(theta_limit),
                           "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                           "Mean scores: " + str(round(mean(scores_no_zero),1)), "Median: " + str(round(median(scores_no_zero),1))))
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    plt.text(0.9, 0.9, hist_text, transform=ax.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=props)
    hist_name = file_name[0:-5] + "_histogram"
    if save:
        # Saves the training data to a numpy file, if save is True!
        file_data = np.array(training_data)
        #np.save(file_name, file_data)  # saves the training data
        with h5py.File(file_name, 'w') as f:
            dset = f.create_dataset("training_data", data=training_data)
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

def NN_simulation(env, NN, theta_lim, score_limit, N_runs, N_steps_per_run, save_data=False, save_hist=False, animation=False, file_name=''):
    """Runs N_runs simulations using the trained Neural Network,
       returns the new training data"""
    scores = []
    choices = []
    training_data = []
    scores_arr = []
    actions = {-step*3: 0, -step*2: 0, -step: 0, step: 0, step*2: 0, step*3: 0}
    accepted_scores = []
    scores_no_zero = []
    div = N_runs/10
    for run in range(N_runs):
        if run % div == 0:
            print("run nr: ", run)
        env.reset(theta_lim)
        score = 0
        runs_memory = []
        actions_memory = [0, 0, 0, 0, 0, 0]
        prev_obs = []
        output = []
        for _ in range(N_steps_per_run):
            if animation:
                env.render()
            if len(prev_obs) == 0:
                action = rnd.randrange(start=lower, stop=upper, step=step)  # If no previous action, a random one is created
                if rnd.random() < 0.5:
                    action = action
                else:
                    action = action * -1
            else:
                action = np.argmax(NN.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])  # argmax returns the index of the output neuron with the largest weight
                force_arr = [-step*3, -step*2, -step, step, step*2, step*3]
                action = force_arr[action]  # Translates the output to force/action applied to the cartpole
            choices.append(action)
            actions[action] += 1
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
                    if data[1] == -step*3:
                        output = np.array([1, 0, 0, 0, 0, 0])
                    if data[1] == -step*2:
                        output = np.array([0, 1, 0, 0, 0, 0])
                    if data[1] == -step:
                        output = np.array([0, 0, 1, 0, 0, 0])
                    if data[1] == step:
                        output = np.array([0, 0, 0, 1, 0, 0])
                    if data[1] == step*2:
                        output = np.array([0, 0, 0, 0, 1, 0])
                    if data[1] == step*3:
                        output = np.array([0, 0, 0, 0, 0, 1])
                    training_data.append([data[0], output])
            if score != 0:
                scores_no_zero.append(score)
            scores.append(score)  # saves the score for that step
        scores_arr.append(score)
    N_bins = int(max(scores))
    fig1, ax = plt.subplots(tight_layout=True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.hist(scores_arr, bins=N_bins)
    font = {'family': 'normal',
            'size': 12}
    plt.rc('font', **font)
    plt.title("NN made training data", fontsize=20)
    hist_text_1 = '\n'.join((' Score limit: ' + str(score_limit), " Theta limit: " + str(theta_lim),
                           "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                           "Mean scores: " + str(round(mean(scores_no_zero), 1)), "Median: " + str(round(median(scores_no_zero), 1))))
    plt.text(0.9, 0.9, hist_text_1, transform=ax.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=props)
    plt.xlabel('scores')
    plt.ylabel('# of runs')
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)

    hist_name_scores = file_name[0:-5] + "_scores_histogram"
    hist_name_force = file_name[0:-5] + "_force_histogram"
    fig2, ax = plt.subplots(tight_layout=True)
    plt.bar(actions.keys(), actions.values(), width=10)
    font = {'family': 'normal',
            'size': 12}
    plt.rc('font', **font)
    plt.title("Force distribution", fontsize=20)
    plt.ylabel("# of times each force is called", fontsize=18)
    plt.xlabel("Force", fontsize=18)
    hist_text_2 = ('Actions to the right: {}%\nActions to the left: {}%'.format((round(np.sum(np.array(choices) >= 0,
                    axis=0) / len(choices)*100)), round(np.sum(np.array(choices) < 0, axis=0) / len(choices)*100)))
    hist_text_2 = hist_text_2 + '\n'.join(('\nScore limit: ' + str(score_limit), " Theta limit: " + str(theta_lim),
                           "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                           "Mean scores: " + str(round(mean(scores_no_zero), 1)), "Median: " + str(round(median(scores_no_zero), 1))))
    plt.text(0.9, 0.9, hist_text_2, transform=ax.transAxes, verticalalignment='top',
             horizontalalignment='right', bbox=props)
    if save_hist:
        plt.show()
        fig1.savefig(hist_name_scores)  # saves the histogram
        fig2.savefig(hist_name_force)
        print("Histogram saved as: ", file_name + "_histogram")
    if save_data:
        # training_data_save = np.array(training_data)
        # np.save(file_name, training_data_save)  # saves the training data
        with h5py.File(file_name, 'w') as f:
            f.create_dataset("training_data", data=training_data)
    print("Trained data saved as: ", file_name)
    return training_data

if __name__ == "__main__":
    """Runs the main code"""
    # random_simulation()
    env = gym()
    theta_limit_1 = 50
    theta_limit_2 = 10
    theta_limit_3 = 15
    theta_limit_sim = 90  # it gets points when it goes above the surface
    N_runs_1 = 10000  # Number of runs per simulation
    N_runs_2 = 20000
    N_runs_3 = 40000
    N_runs_sim = 1000
    N_steps_per_run_1 = 100  # Maximum number of steps per run for each simulation
    N_steps_per_run_2 = 100
    N_steps_per_run_3 = 100
    N_steps_per_run_sim = 200
    score_limit_1 = 40  # minimum score limit
    score_limit_2 = 5
    score_limit_3 = 10
    score_limit_sim = 30
    input_size = 6
    output_size = 6

    """Saving the paramaters used for the training data"""
    """
    print("Saving paramaters as a txt file... Saved!")
    param_file = open("parameters_data_v5_3", "w")
    parameters_info_text = "Paramaters info\nTheta limit 1: " + str(theta_limit_1) + "\n  Score limit 1: " + str(score_limit_1) \
                           + "\n  Number of runs 1: " + str(N_runs_1) + "\n  Number of steps per run 1: " + str(N_steps_per_run_1)\
                           + "\n\n  Theta limit 2: " + str(theta_limit_2) + "\n  Score limit 2: " + str(score_limit_2) \
                           + "\n  Number of runs 2: " + str(N_runs_2) + "\n  Number of steps per run 2: " + str(N_steps_per_run_2)\
                           + "\n\n  Theta limit 3: " + str(theta_limit_3) + "\n  Score limit 3: " + str(score_limit_3) \
                           + "\n  Number of runs 3: " + str(N_runs_3) + "\n  Number of steps per run 3: " + str(N_steps_per_run_3)\
                           + "\n\nNN simulation\n" + "  Score limit simulation: " + str(score_limit_sim) + "\n  Number of runs sim: " + str(N_runs_sim) \
                           + "\n  Number of steps per run sim: " + str(N_steps_per_run_sim) + "\n\nModel parameters\n  " + "Learning rate: " \
                           + str(LR) + "\n  Keep probability: " + str(epoch) + "\n  epoch: " + str(epoch) \
                           + "\n\nForce range: [" + str(-step*3) + ", " + str(-step*2) + ", " + str(-step) + ", " \
                           + str(step) + ", " + str(step*2) + ", " + str(step*3) + "]" + "\nForce step: " + str(step)
    param_file.write(parameters_info_text)
    param_file.close()
    """

    """Load old models (make sure the file is in the same folder as the code!"""
    # model_1 = load_model('cartpole_model_v5_1_1.tflearn', input_size, output_size)
    # model_2 = load_model('cartpole_model_v5_3_2.tflearn', input_size, output_size)
    # model_3 = load_model('cartpole_model_v5_2_3.tflearn', input_size, output_size)


    """Create new training_data (randomly!)"""
    """
    training_data_1 = create_training_data(env, theta_limit_1, score_limit=score_limit_1, save=True, file_name='training_data_v5_3_1.hdf5',
                                          N_runs=N_runs_1, N_steps_per_run=N_steps_per_run_1)
    
    training_data_2 = create_training_data(env, theta_limit_2, score_limit=score_limit_2, save=True, file_name='training_data_v5_3_2.npy',
                                           N_runs=N_runs_2, N_steps_per_run=N_steps_per_run_2)
   
    training_data_3 = create_training_data(env, theta_limit_3, score_limit=score_limit_3, save=True, file_name='training_data_v5_2_3.npy',
                                           N_runs=N_runs_3, N_steps_per_run=N_steps_per_run_3)
    """
    # print("training data 1 size: ", len(training_data_1))
    # print("training data 2 size: ", len(training_data_2))
    # print("training data 3 size: ", len(training_data_3))

    """Load old training_data_#"""
    """Creates or trains new models based on the training data"""
    """
    with h5py.File('training_data_v5_1_1.hdf5', 'r') as f:
        data_set = f['training_data']
        training_data_1 = data_set[:]
        
        model_1 = train_model(training_data_1, model=False, new_model=True, save=True,
                              file_name='cartpole_model_v5_1_1.tflearn')
    """

    """Creates or trains new models based on the training data"""
    # model_1 = train_model(training_data_1, model=False, new_model=True, save=True, file_name='cartpole_model_v5_3_1.tflearn')
    # model_2 = train_model(training_data_2, model=model_1, new_model=False, save=True, file_name='cartpole_model_v5_3_2.tflearn')
    # model_3 = train_model(training_data_3, model=False, new_model=True, save=True, file_name='cartpole_model_v5_2_1.tflearn')

    """Run new simulation with existing model and saves the new training data"""
    training_data_1_trained = NN_simulation(env, model_2, theta_limit_sim, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                                            save_data=False, save_hist=False, animation=True, file_name='training_data_v5_3_1_trained.hdf5')
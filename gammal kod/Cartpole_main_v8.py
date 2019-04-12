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
from gym_kod_v8 import CartPoleEnv as gym
import random as rnd
import numpy as np
import h5py
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# "global variables"
LR = 1e-3  # Learning Rate
keep_prob = 0.8
epoch = 1

def Neural_Network(input_size, output_size):
    """Creates a Neural network, returns a neural network"""

    network = input_data(shape=[None, input_size, 1], name='input')

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

def create_training_data(env, theta_limit, force_vector, save_plots=False, animation=False, N_runs=None, N_steps_per_run=None, score_limit=None, file_name=''):
    """Creates and saves training data that gets score above score_limit """
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    scores_no_zero = []
    accepted_scores = []  # saves the accepted scores
    scores_arr = []
    pole_length = []
    pole_mass = []
    mass_cart = []
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
            if len(prev_observation) == 0:
                action = rnd.choice([force_vector[0], force_vector[1], force_vector[2], force_vector[4], force_vector[5], force_vector[6]])
            else:
                action = rnd.choice(force_vector)
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
                    if data[1] == force_vector[0]:
                        output = np.array([1, 0, 0, 0, 0, 0, 0])
                    if data[1] == force_vector[1]:
                        output = np.array([0, 1, 0, 0, 0, 0, 0])
                    if data[1] == force_vector[2]:
                        output = np.array([0, 0, 1, 0, 0, 0, 0])
                    if data[1] == force_vector[3]:
                        output = np.array([0, 0, 0, 1, 0, 0, 0])
                    if data[1] == force_vector[4]:
                        output = np.array([0, 0, 0, 0, 1, 0, 0])
                    if data[1] == force_vector[5]:
                        output = np.array([0, 0, 0, 0, 0, 1, 0])
                    if data[1] == force_vector[6]:
                        output = np.array([0, 0, 0, 0, 0, 0, 1])
                    training_data.append([data[0], output])
            if score != 0:
                scores_no_zero.append(score)
            scores.append(score)  # saves the score for that step
        scores_arr.append(score)
        pole_length.append(observation[4])
        pole_mass.append(observation[5])
        mass_cart.append(observation[6])

    if save_plots:
        # 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_trisurf(pole_length, pole_mass, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Pole mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_1"))
        plt.savefig(name_3d_file)  # saves the 3D plot
        plt.close(fig)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_length, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_2"))
        plt.savefig(name_3d_file)  # saves the 3D plot
        plt.close(fig)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_mass, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole mass')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_3"))
        plt.savefig(name_3d_file)  # saves the 3D plot

        plt.close(fig)
        # Plots the histograms
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
                               "Mean scores: " + str(round(mean(scores_no_zero), 1)),
                               "Median: " + str(round(median(scores_no_zero), 1))))
        plt.xlabel('scores')
        plt.ylabel('# of runs')
        ax.yaxis.label.set_size(20)
        ax.xaxis.label.set_size(20)
        plt.text(0.9, 0.9, hist_text, transform=ax.transAxes, verticalalignment='top',
                 horizontalalignment='right', bbox=props)
        hist_name = file_name[0:-5] + "_histogram"
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

def train_model(training_data=None, model=None, save=False, file_name=''):
    """Trains the neural network model with training_data"""
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)  # Observation variables! From training data
    y = [i[1] for i in training_data]  # All forces

    if model is None:  # Creates a new model if no model is entered as attribute
        model = Neural_Network(input_size=len(X[0]), output_size=len(y[0]))
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True,
                  run_id='openai_learning')
        if save is True:
            # Save a model
            model.save(file_name)
            print("Model saved as: ", file_name)
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

def NN_simulation(env, NN, theta_lim, force_vector, score_limit, N_runs, N_steps_per_run, save_data=False, save_plots=False, animation=False, file_name=''):
    """Runs N_runs simulations using the trained Neural Network,
       returns the new training data"""
    scores = []
    choices = []
    training_data = []
    scores_arr = []
    pole_length = []
    pole_mass = []
    mass_cart = []
    actions = {force_vector[0]: 0, force_vector[1]: 0, force_vector[2]: 0, force_vector[3]: 0, force_vector[4]: 0, force_vector[5]: 0, force_vector[6]: 0}
    accepted_scores = []
    scores_no_zero = []
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
                action = rnd.choice([force_vector[0], force_vector[1], force_vector[2], force_vector[4], force_vector[5], force_vector[6]])
            else:
                action = np.argmax(NN.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])  # argmax returns the index of the output neuron with the largest weight
                action = force_vector[action]  # Translates the output to force/action applied to the cartpole
            choices.append(action)
            actions[action] += 1
            observation, reward, done, info = env.step(action)
            prev_obs = observation
            runs_memory.append([observation, action])
            score += reward
            if done:
                break
            if score >= score_limit:
                accepted_scores.append(score)
                for data in runs_memory:
                    """Translates the force/ action to an normalized array"""
                    if data[1] == force_vector[0]:
                        output = np.array([1, 0, 0, 0, 0, 0, 0])
                    if data[1] == force_vector[1]:
                        output = np.array([0, 1, 0, 0, 0, 0, 0])
                    if data[1] == force_vector[2]:
                        output = np.array([0, 0, 1, 0, 0, 0, 0])
                    if data[1] == force_vector[3]:
                        output = np.array([0, 0, 0, 1, 0, 0, 0])
                    if data[1] == force_vector[4]:
                        output = np.array([0, 0, 0, 0, 1, 0, 0])
                    if data[1] == force_vector[5]:
                        output = np.array([0, 0, 0, 0, 0, 1, 0])
                    if data[1] == force_vector[6]:
                        output = np.array([0, 0, 0, 0, 0, 0, 1])
                    training_data.append([data[0], output])
            if score != 0:
                scores_no_zero.append(score)
            scores.append(score)  # saves the score for that step
        scores_arr.append(score)
        pole_length.append(observation[4])
        pole_mass.append(observation[5])
        mass_cart.append(observation[6])
    scores_arr = np.array(scores_arr)
    pole_length = np.array(pole_length)
    pole_mass = np.array(pole_mass)

    if save_plots:
        # 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_trisurf(pole_length, pole_mass, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Pole mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_1"))
        plt.savefig(name_3d_file)  # saves the 3D plot

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_length, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_2"))
        plt.savefig(name_3d_file)  # saves the 3D plot

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_mass, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole mass')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title("Random\nCartPole variables vs Scores")
        name_3d_file = "_".join((file_name[0:-5], "random_3D_plot_3"))
        plt.savefig(name_3d_file)  # saves the 3D plot

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
                                 "Mean scores: " + str(round(mean(scores_no_zero), 1)),
                                 "Median: " + str(round(median(scores_no_zero), 1))))
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
        hist_text_2 = (
            'Actions to the right: {}%\nActions to the left: {}%'.format((round(np.sum(np.array(choices) >= 0,
                                                                                       axis=0) / len(choices) * 100)),
                                                                         round(np.sum(np.array(choices) < 0,
                                                                                      axis=0) / len(choices) * 100)))
        hist_text_2 = hist_text_2 + '\n'.join(('\nScore limit: ' + str(score_limit), " Theta limit: " + str(theta_lim),
                                               "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                                               "Mean scores: " + str(round(mean(scores_no_zero), 1)),
                                               "Median: " + str(round(median(scores_no_zero), 1))))
        plt.text(0.9, 0.9, hist_text_2, transform=ax.transAxes, verticalalignment='top',
                 horizontalalignment='right', bbox=props)
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

    file_name = "CartPole v8_1_1"
    Info_Text = ""
    Create_new_model = False
    Load_and_run_old_model = True
    force_vector = [-150, -70, -20, 0, 20, 70, 150]
    force_vector_train = [-50, -30, -20, 0, 20, 30, 50]

    theta_limit_1 = 90
    score_limit_1 = 400  # minimum score limit
    N_runs_1 = 100000  # Number of runs per
    N_steps_per_run_1 = 500  # Maximum number of steps per run for each simulation

    theta_limit_sim = 90  # it gets points when it goes above the surface
    score_limit_sim = 2000
    N_runs_sim = 200
    N_steps_per_run_sim = 1000

    input_size = 7
    output_size = 7

    env = gym()  # Creates the enviroment

    if Create_new_model:
        """Saving the paramaters used for the training data"""
        param_file = open("_".join((file_name, "parameters")), "w")
        parameters_info_text = "Info: \n" + Info_Text \
                               + "Paramaters info\nTheta limit 1: " + str(theta_limit_1) + "\n  Score limit 1: " + str(score_limit_1) \
                               + "\n  Number of runs 1: " + str(N_runs_1) + "\n  Number of steps per run 1: " + str(N_steps_per_run_1)\
                               + "\n\nNN simulation\n" + "  Score limit simulation: " + str(score_limit_sim) + "\n  Number of runs sim: " + str(N_runs_sim) \
                               + "\n  Number of steps per run sim: " + str(N_steps_per_run_sim) + "\n\nModel parameters\n  " + "Learning rate: " \
                               + str(LR) + "\n  Keep probability: " + str(epoch) + "\n  epoch: " + str(epoch) \
                               + "\nForce vector:" + str(force_vector)
        print("Saving paramaters as a txt file... Saved!")
        param_file.write(parameters_info_text)
        param_file.close()

        """Create new training_data (randomly!)"""
        training_data = create_training_data(env, theta_limit_1, force_vector, score_limit=score_limit_1, save_plots=True,
                                               file_name="_".join((file_name, "training_data.hdf5")), N_runs=N_runs_1,
                                               animation=False, N_steps_per_run=N_steps_per_run_1)

        """Creates or trains new models based on the training data"""
        model = train_model(training_data, model=None, save=True, file_name=".".join((file_name, "tflearn")))

        """Run new simulation with existing model and saves the new training data"""
        training_data_1_trained = NN_simulation(env, model, theta_limit_sim, force_vector, score_limit_sim,
                                                N_runs_sim, N_steps_per_run_sim,
                                                save_data=False, save_plots=True, animation=False,
                                                file_name="_".join((file_name, "simulation.hdf5")))

        NN_simulation(env, model, theta_limit_sim, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                      save_data=False, save_plots=False, animation=True,
                      file_name="_".join((file_name, "simulation.hdf5")))

    if Load_and_run_old_model:
        """Load old models (make sure the file is in the same folder as the code!"""
        model = load_model(".".join((file_name, "tflearn")), input_size, output_size)

        NN_simulation(env, model, theta_limit_sim, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                      save_data=False, save_plots=False, animation=True,
                      file_name="_".join((file_name, "simulation.hdf5")))




    """Load old training_data_#"""
    """Creates or trains new models based on the training data"""
    """
    with h5py.File('training_data_v7_2_1_trained.hdf5', 'r') as f:
        data_set = f['training_data']
        training_data_1 = data_set[:]

        model_1_trained = train_model(training_data_1, model=model, new_model=False, save=True,
                              file_name='cartpole_model_v7_2_1_trained.tflearn')
    """


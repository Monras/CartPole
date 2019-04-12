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

def simulation(env, NN, theta_lim, force_vector, score_limit, N_runs, N_steps_per_run, save_data=False,
                  save_plots=False, animation=False, create_training_data=False, run_simulation=False, file_name=''):
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    scores_no_zero = []
    accepted_scores = []  # saves the accepted scores
    scores_arr = []
    pole_length = []
    pole_mass = []
    mass_cart = []
    choices = []
    actions = {force_vector[0]: 0, force_vector[1]: 0, force_vector[2]: 0, force_vector[3]: 0, force_vector[4]: 0,
               force_vector[5]: 0, force_vector[6]: 0}
    div = N_runs / 10
    for run in range(N_runs):
        if run % div == 0:
            print("run nr: ", run)
        env.reset(theta_lim)  # resets the simulation, does the first step
        score = 0
        runs_memory = []
        prev_obs = []
        output = []
        for _ in range(N_steps_per_run):
            if animation:
                env.render()  # displays the simulation
            if len(prev_obs) == 0:
                """Creates first action randomly"""
                action = rnd.choice(
                    [force_vector[0], force_vector[1], force_vector[2], force_vector[4], force_vector[5],
                     force_vector[6]])
                choices.append(action)
                actions[action] += 1
            elif len(prev_obs) > 0:
                if create_training_data:
                    """If create training data is True, choose the action randomly"""
                    action = rnd.choice(force_vector)
                elif run_simulation:
                    """If run simulation is True, predict the action using the previous observation"""
                    action = np.argmax(NN.predict(prev_obs.reshape(-1, len(prev_obs), 1))[
                                       0])  # argmax returns the index of the output neuron with the largest weight
                    action = force_vector[action]  # Translates the output to force/action applied to the cartpole
                    choices.append(action)
                    actions[action] += 1
            else:
                ValueError("Something went wrong calculating the observations and actions!")

            observation, reward, done, info = env.step(action)  # Calculates the next step, with the new action

            if len(prev_obs) > 0:
                runs_memory.append([prev_obs, action])  # saves the action and the previous observation
            prev_obs = observation  # saves the current observation as previous observation
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
        # Adds the runs static variables to respective array
        pole_length.append(observation[4])
        pole_mass.append(observation[5])
        mass_cart.append(observation[6])

    # Change the arrays to numpy arrays
    scores_arr = np.array(scores_arr)
    pole_length = np.array(pole_length)
    pole_mass = np.array(pole_mass)

    if save_plots:
        if create_training_data:
            title_3D = "Randomly created actions\nCartPole variables vs Scores"
            file_name_3D = "random_3D_plot"
            hist_scores_title = "Randomly created training data"

        if run_simulation:
            title_3D = "Using machine learning\nCartPole variables vs Scores"
            file_name_3D = "3D_plot"
            hist_scores_title = "Training data created using machine learning"

        else:
            ValueError("Oops! something went wrong in save_plots!")

        # 3D plots
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_trisurf(pole_length, pole_mass, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Pole mass')
        ax.set_zlabel('Scores')
        ax.set_title(title_3D)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "1"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        plt.close(fig)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_length, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title(title_3D)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "2"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        plt.close(fig)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_mass, mass_cart, scores_arr, cmap=cm.coolwarm)
        ax.set_xlabel('Pole mass')
        ax.set_ylabel('Cart mass')
        ax.set_zlabel('Scores')
        ax.set_title(title_3D)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "3"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        plt.close(fig)

        # Plots the number of runs vs scores histogram
        N_bins = int(max(scores))
        fig1, ax = plt.subplots(tight_layout=True)
        ax.set_facecolor([0.95, 0.95, 0.95])
        props = dict(boxstyle='round', alpha=0.5, facecolor='wheat')
        plt.hist(scores_arr, bins=N_bins)
        font = {'family': 'normal',
                'size': 12}
        plt.rc('font', **font)
        plt.title(hist_scores_title, fontsize=20)
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
        fig1.savefig(hist_name_scores)  # saves the histogram

        if run_simulation:
            # Plots force histogram
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
            fig2.savefig(hist_name_force)
        plt.show()
        print("3D plots saved as: ", file_name + file_name_3D)
        print("Histogram saved as: ", file_name + "_histogram")
    if save_data:
        # training_data_save = np.array(training_data)
        # np.save(file_name, training_data_save)  # saves the training data
        with h5py.File(file_name, 'w') as f:
            f.create_dataset("training_data", data=training_data)
        print("Trained data saved as: ", file_name)
    return training_data

def train_model(training_data=None, model=None, save=False, file_name=''):
    """Trains the neural network model with training_data"""
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),
                                                        1)  # Observation variables! From training data
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
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=500, show_metric=True,
                  run_id='openai_learning')
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


if __name__ == "__main__":
    """Runs the main code"""

    file_name = "CartPole v8_1_1"
    Info_Text = ""
    Create_new_model = True
    Load_and_run_old_model = False
    force_vector = [-300, -200, -100, 0, 100, 200, 300]
    # force_vector_train = [-50, -30, -20, 0, 20, 30, 50]

    theta_limit_1 = 90
    score_limit_1 = 300  # minimum score limit
    N_runs_1 = 10000  # Number of runs per
    N_steps_per_run_1 = 500  # Maximum number of steps per run for each simulation

    theta_limit_sim = 90  # it gets points when it goes above the surface
    score_limit_sim = 2000
    N_runs_sim = 200
    N_steps_per_run_sim = 500

    input_size = 7
    output_size = 7

    env = gym()  # Creates the enviroment

    if Create_new_model:
        """Saving the paramaters used for the training data"""
        param_file = open("_".join((file_name, "parameters")), "w")
        parameters_info_text = "Info: \n" + Info_Text \
                               + "Paramaters info\nTheta limit 1: " + str(theta_limit_1) + "\n  Score limit 1: " + str(
            score_limit_1) \
                               + "\n  Number of runs 1: " + str(N_runs_1) + "\n  Number of steps per run 1: " + str(
            N_steps_per_run_1) \
                               + "\n\nNN simulation\n" + "  Score limit simulation: " + str(
            score_limit_sim) + "\n  Number of runs sim: " + str(N_runs_sim) \
                               + "\n  Number of steps per run sim: " + str(
            N_steps_per_run_sim) + "\n\nModel parameters\n  " + "Learning rate: " \
                               + str(LR) + "\n  Keep probability: " + str(epoch) + "\n  epoch: " + str(epoch) \
                               + "\nForce vector:" + str(force_vector)
        print("Saving paramaters as a txt file... Saved!")
        param_file.write(parameters_info_text)
        param_file.close()

        """Create new training_data (randomly!)"""
        model = None
        training_data = simulation(env, model, theta_limit_1, force_vector, score_limit_1, N_runs_1, N_steps_per_run_1,
                                             save_plots=True, animation=False,
                                             create_training_data=True, run_simulation=False,
                                             file_name="_".join((file_name, "training_data.hdf5")))

        """Creates or trains new models based on the training data"""
        model = train_model(training_data, model=None, save=True, file_name=".".join((file_name, "tflearn")))

        """Run new simulation with existing model and saves the new training data"""
        training_data_1_trained = simulation(env, model, theta_limit_sim, force_vector, score_limit_sim,
                                                N_runs_sim, N_steps_per_run_sim,
                                                save_data=False, save_plots=True, animation=False,
                                                create_training_data=False, run_simulation=True,
                                                file_name="_".join((file_name, "simulation.hdf5")))

        simulation(env, model, theta_limit_sim, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                      save_data=False, save_plots=False, animation=True,
                      create_training_data=False, run_simulation=True,
                      file_name="_".join((file_name, "simulation.hdf5")))

    if Load_and_run_old_model:
        """Load old models (make sure the file is in the same folder as the code!"""
        model = load_model(".".join((file_name, "tflearn")), input_size, output_size)

        training_data = simulation(env, model, theta_limit_sim, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                      save_data=False, save_plots=True, animation=False,
                   create_training_data=False, run_simulation=True,
                      file_name="_".join((file_name, "simulation.hdf5")))

        simulation(env, model, theta_limit_sim, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                   save_data=False, save_plots=False, animation=True,
                   create_training_data=False, run_simulation=True,
                   file_name="_".join((file_name, "simulation.hdf5")))

    """
    Lägg till att du når self.Mass_cart genom env objektet istället. 

    Gör en if sats så vi kan träna data och köra simulationen med olika villkort på en körning.

    """

    """Load old training_data_#"""
    """Creates or trains new models based on the training data"""
    """
    with h5py.File('training_data_v7_2_1_trained.hdf5', 'r') as f:
        data_set = f['training_data']
        training_data_1 = data_set[:]

        model_1_trained = train_model(training_data_1, model=model, new_model=False, save=True,
                              file_name='cartpole_model_v7_2_1_trained.tflearn')
    """


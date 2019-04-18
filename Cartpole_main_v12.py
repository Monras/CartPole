
' '# By Måns Rasmussen and Gustaf Holte

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
from gym_kod_v12 import CartPoleEnv as gym
from gym import wrappers
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
    """
    Tesnorflow verbose:
    0: Loss & Metric (Best speed).
    1: Loss, Metric & Gradients.
    2: Loss, Metric, Gradients & Weights.
    3: Loss, Metric, Gradients, Weights, Activations & Sparsity (Best Visualization).
    """
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, keep_prob)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, keep_prob)

    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                         name='targets')  # Creates a regression, optimizes the network using back-propagation
    model = tflearn.DNN(network)  # Creates the neural network

    return model

def simulation(env, model, force_vector, score_limit, N_runs, N_steps_per_run, save_data=False,
                  save_plots=False, animation=False, create_training_data=False, run_simulation=False, file_name=''):
    training_data = []  # saves the selected training data
    accepted_scores = []  # saves the accepted scores
    scores = []
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
        env.reset()  # resets the simulation, does the first step
        score = 0
        runs_memory = []
        prev_obs = []
        output = []
        for _ in range(N_steps_per_run):
            if animation:
                env.render()  # displays the simulation
            # Creates new training data randomly
            if create_training_data:
                if len(prev_obs) == 0:
                    """Creates first action randomly, without the zero!"""
                    action = rnd.choice(
                        [force_vector[0], force_vector[1], force_vector[2], force_vector[4], force_vector[5],
                         force_vector[6]])
                else:
                    action = rnd.choice(force_vector)
                choices.append(action)
                actions[action] += 1
                new_observation, reward, done, info = env.step(action)  # Calculates the next step, with the new action
                if len(prev_obs) > 0:
                    runs_memory.append([prev_obs, action])
                prev_obs = new_observation  # saves the current observation as previous observation
            # Runs the simulation using NN commands for actions
            if run_simulation:
                """If run simulation is True, predict the action using the previous observation"""
                if len(prev_obs) == 0:
                    """Creates first action randomly, without the zero!"""
                    action = rnd.choice(
                        [force_vector[0], force_vector[1], force_vector[2], force_vector[4], force_vector[5],
                         force_vector[6]])
                else:
                    # argmax returns the index of the output neuron with the largest weight
                    action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
                    action = force_vector[action]  # Translates the output to force/action applied to the cartpole
                choices.append(action)
                actions[action] += 1
                new_observation, reward, done, info = env.step(action)  # Calculates the next step, with the new action
                prev_obs = new_observation  # saves the current observation as previous observation
                runs_memory.append([new_observation, action])
            else:
                ValueError("Something went wrong calculating the observations and actions!")
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
        scores.append(score)
        # Adds the runs static variables to respective array
        pole_length.append(new_observation[4])
        pole_mass.append(new_observation[5])
        mass_cart.append(new_observation[6])

    # Change the arrays to numpy arrays
    scores = np.array(scores)
    pole_length = np.array(pole_length)
    pole_mass = np.array(pole_mass)
    mass_cart = np.array(mass_cart)

    if save_plots:
        if create_training_data:
            title_3D = "Randomly created actions\nCartPole variables vs Scores"
            file_name_3D = "random_3D_plot"
            hist_scores_title = "Randomly created training data"

        if run_simulation:
            title_3D = "Using predicted actions\nCartPole variables vs Scores"
            file_name_3D = "3D_plot"
            hist_scores_title = "Training data created using machine learning"

        else:
            ValueError("Oops! something went wrong in save_plots!")

        if save_data:
            # training_data_save = np.array(training_data)
            # np.save(file_name, training_data_save)  # saves the training data
            with h5py.File(file_name, 'w') as f:
                f.create_dataset("training_data", data=training_data)
            print("Trained data saved as: ", file_name)

        # 3D plots
        font = {'family': 'normal',
                'size': 11}
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #plt.rc('font', **font)
        # Plot the surface.
        surf = ax.plot_trisurf(pole_length, pole_mass, scores, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length', fontsize=16)
        ax.set_ylabel('Pole mass', fontsize=16)
        ax.set_zlabel('Scores', fontsize=16)
        ax.set_title(title_3D, fontsize=20)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "1"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        # plt.close(fig)  # Closes the 3D plot

        # 3D plot pole length vs mass cart
        #plt.rc('font', **font)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_length, mass_cart, scores, cmap=cm.coolwarm)
        ax.set_xlabel('Pole length', fontsize=16)
        ax.set_ylabel('Cart mass', fontsize=16)
        ax.set_zlabel('Scores', fontsize=16)
        ax.set_title(title_3D, fontsize=20)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "2"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        # plt.close(fig)  # Closes the 3D plot
        #plt.rc('font', **font)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(pole_mass, mass_cart, scores, cmap=cm.coolwarm)
        ax.set_xlabel('Pole mass', fontsize=16)
        ax.set_ylabel('Cart mass', fontsize=16)
        ax.set_zlabel('Scores', fontsize=16)
        ax.set_title(title_3D, fontsize=20)
        name_3d_file = "_".join((file_name[0:-5], "_".join((file_name_3D, "3"))))
        plt.savefig(name_3d_file)  # saves the 3D plot
        # plt.close(fig)  # Closes the 3D plot

        """"""
        # Plots the number of runs vs scores histogram
        N_bins = int(max(scores))
        fig1, ax = plt.subplots(tight_layout=True)
        ax.set_facecolor([0.95, 0.95, 0.95])
        props = dict(boxstyle='round', alpha=0.5, facecolor='wheat')
        plt.hist(scores, bins=N_bins)
        font = {'family': 'normal',
                'size': 12}
        plt.rc('font', **font)
        plt.title(hist_scores_title, fontsize=20)
        hist_text_1 = '\n'.join((' Score limit: ' + str(score_limit),
                                 "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                                 "Mean scores: " + str(round(mean(scores), 1)),
                                 "Median: " + str(round(median(scores), 1))))
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
            hist_text_2 = hist_text_2 + '\n'.join(('\nScore limit: ' + str(score_limit),
                                                   "N_runs: " + str(N_runs), "N_steps_per_run: " + str(N_steps_per_run),
                                                   "Mean scores: " + str(round(mean(scores), 1)),
                                                   "Median: " + str(round(median(scores), 1))))
            plt.text(0.9, 0.9, hist_text_2, transform=ax.transAxes, verticalalignment='top',
                     horizontalalignment='right', bbox=props)
            fig2.savefig(hist_name_force)
        plt.show()
        print("3D plots saved as: ", file_name + file_name_3D)
        print("Histogram saved as: ", file_name + "_histogram")

    return training_data

def train_model(training_data=None, model=None, save=False, file_name=''):
    """Trains the neural network model with training_data"""
    # Observation variables! From training data transform row to column. All variables are in a row after each other
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]  # All forces
    if model is None:  # Creates a new model if no model is entered as attribute
        model = Neural_Network(input_size=len(X[0]), output_size=len(y[0]))
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=50000, show_metric=True,
                  run_id='openai_learning')
        if save is True:
            # Save a model
            model.save(file_name)
            print("Model saved as: ", file_name)
        return model
    elif model is not None:
        model.fit({'input': X}, {'targets': y}, n_epoch=epoch, snapshot_step=50000, show_metric=True,
                  run_id='openai_learning')
        if save is True:
            # Save a model
            model.save(file_name)
            print("Model saved as: ", file_name)
        return model
    else:
        raise ValueError("Something went wrong when training the model")

def see_training(env, training_data, N_runs, N_steps_per_run):
    div = N_runs / 10
    j = 0
    for run in range(N_runs):
        env.reset()
        print("\nrun nr:\n ", run)

        for i in range(N_steps_per_run):

            action = training_data[j][1]
            action = np.argmax(action)
            observation = training_data[j][0]
            #print("action: ", action)
            #print("observation: ", observation)
            new_observation, reward, done, info = env.step(action, observation)  # Calculates the next step, with the new action
            env.render()  # displays the simulation
            j += 1
            if done:
                break
        j += 1
def load_model(file_name, input_size, output_size):
    """Loads and returns a NN model"""
    print("Loading file: ", file_name)
    model = Neural_Network(input_size=input_size, output_size=output_size)
    model.load(file_name)
    return model


if __name__ == "__main__":
    """Runs the main code"""

    file_name = "CartPole v10_5"  # For saving!
    open_file = "CartPole v10_3_1"
    Info_Text = ""
    Create_new_model = True
    create_new_model_on_old_data = False
    Load_and_train_old_model = False
    load_model_and_run = False

    force_vector = [-300, -200, -100, 0, 100, 200, 300]
    # force_vector_weak = [-50, -25, -10, 0, 10, 25, 50]
    # force_vector = force_vector_weak

    score_limit_1 = 600  # minimum score limit
    N_runs_1 = 10000  # Number of runs per
    N_steps_per_run_1 = 500  # Maximum number of steps per run for each simulation

    score_limit_sim = 2000
    N_runs_sim = 500
    N_steps_per_run_sim = 100

    input_size = 7
    output_size = 7

    env = gym()  # Creates the enviroment
    # env = wrappers.Monitor(env, 'output_movie', force=True)  # saves the animation to an mp4 movie

    if Create_new_model:
        """Saving the paramaters used for the training data"""
        param_file = open("_".join((file_name, "parameters")), "w")
        parameters_info_text = "Info: \n" + Info_Text \
                               + "Paramaters info\n" + "\n  Score limit 1: " + str(
            score_limit_1) \
                               + "\n  Number of runs 1: " + str(N_runs_1) + "\n  Number of steps per run 1: " + str(
            N_steps_per_run_1) \
                               + "\n\nNN simulation\n" + "  Score limit simulation: " + str(
            score_limit_sim) + "\n  Number of runs sim: " + str(N_runs_sim) \
                               + "\n  Number of steps per run sim: " + str(
            N_steps_per_run_sim) + "\n\nModel parameters\n  " + "Learning rate: " \
                               + str(LR) + "\n  Keep probability: " + str(keep_prob) + "\n  epoch: " + str(epoch) \
                               + "\nForce vector:" + str(force_vector)
        print("Saving paramaters as a txt file... Saved!")
        param_file.write(parameters_info_text)
        param_file.close()

        """Create new training_data (randomly!)"""
        model = None
        training_data = simulation(env, model, force_vector, score_limit_1, N_runs_1, N_steps_per_run_1,
                                             save_plots=True, animation=False, save_data=True,
                                             create_training_data=True, run_simulation=False,
                                             file_name="_".join((file_name, "training_data.hdf5")))

        # see_training(env, training_data, 2, N_steps_per_run_1)

        """Creates or trains new models based on the training data"""
        model = train_model(training_data, model=None, save=True, file_name=".".join((file_name, "tflearn")))

        """Run new simulation with existing model and saves the new training data"""
        training_data_1_trained = simulation(env, model, force_vector, score_limit_sim,
                                                N_runs_sim, N_steps_per_run_sim,
                                                save_data=False, save_plots=True, animation=False,
                                                create_training_data=False, run_simulation=True,
                                                file_name="_".join((file_name, "simulation.hdf5")))

        simulation(env, model, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                      save_data=False, save_plots=False, animation=True,
                      create_training_data=False, run_simulation=True,
                      file_name="_".join((file_name, "simulation.hdf5")))

    if load_model_and_run:
        model = load_model(".".join((open_file, "tflearn")), input_size, output_size)
        """Load old models (make sure the file is in the same folder as the code!"""
        training_data = simulation(env, model, force_vector, score_limit_sim, N_runs_sim,
                                   N_steps_per_run_sim,
                                   save_data=False, save_plots=True, animation=False,
                                   create_training_data=False, run_simulation=True,
                                   file_name="_".join((file_name, "simulation_trained.hdf5")))

    if create_new_model_on_old_data:
        """Load old training_data_#"""
        """Creates or trains new models based on the training data"""
        with h5py.File("_".join((open_file, "training_data.hdf5")), 'r') as f:
            data_set = f['training_data']
            training_data = data_set[:]
            model = train_model(training_data, model=None, save=True,
                                  file_name=".".join((file_name, "tflearn")))

        training_data_trained = simulation(env, model, force_vector, score_limit_sim, N_runs_sim,
                                   N_steps_per_run_sim,
                                   save_data=True, save_plots=True, animation=False,
                                   create_training_data=False, run_simulation=True,
                                   file_name="_".join((file_name, "simulation_trained.hdf5")))

        simulation(env, model, force_vector, score_limit_sim, N_runs_sim, N_steps_per_run_sim,
                   save_data=False, save_plots=False, animation=True,
                   create_training_data=False, run_simulation=True,
                   file_name="_".join((file_name, "simulation.hdf5")))

    if Load_and_train_old_model:
        model = load_model(".".join((open_file, "tflearn")), input_size, output_size)
        """Creates or trains new models based on the training data"""
        with h5py.File("_".join((open_file, "simulation_trained.hdf5")), 'r') as f:
            data_set = f['training_data']
            training_data = data_set[:]
            model_train = train_model(training_data, model=model, save=True,
                                      file_name=".".join((file_name, "trained_tflearn")))

        """Load old models (make sure the file is in the same folder as the code!"""
        training_data = simulation(env, model, force_vector, score_limit_sim, N_runs_sim,
                                   N_steps_per_run_sim,
                                   save_data=False, save_plots=True, animation=True,
                                   create_training_data=False, run_simulation=True,
                                   file_name="_".join((file_name, "simulation_trained.hdf5")))

    """
    Lägg till att du når self.Mass_cart genom env objektet istället. 

    Gör en if sats så vi kan träna data och köra simulationen med olika villkort på en körning.

    """




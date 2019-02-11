# By Måns Rasmussen and Gustaf Holte
"""Simulates an inverted pendulum and balance it using Neural Network coding"""
# imports packages
from gym_kod import CartPoleEnv as gym
import random as rnd
import numpy as np
from statistics import median, mean

# Variables
N_runs= 20  # Number of different runs of cartpole simulations
N_steps_per_run = 100  # Maximum number of steps per run for each simulation
score_limit = 50  # minimum score limit

def create_training_data():
    """Creates and saves training data that gets score above score_limit """
    training_data = []  # saves the selected training data
    scores = []  # saves the scores
    accepted_scores = []  # saves the accepted scores

    for _ in range(N_runs):
        score = 0
        runs_memory = []  # saves the runs
        prev_observation = []
        for _ in range(N_steps_per_run):
            action = rnd.randrange(0,2)  # creates a random number: 0 or 1
            observation, reward, done, info = env.step(action)  # updates the observation with given action

            if len(prev_observation) > 0:
                """saves the next observation and action"""
                runs_memory.append([observation, action])  # saves the action that gave that observation
                score += reward  # updates the score as long as done is not False.
                if done:
                    break

            if score >= score_limt:
                accepted_scores.append(score)
                for data in runs_memory:
                    """Checks if the action is 1 or 0"""
                    if data[1] == 1:
                        output = [0, 1]
                    elif data[1] == 0:
                        output = [1, 0]

                    training_data.append([data[0], output])

            env.render()  # updates the animation
            env.reset()
            scores.append(score)  # saves the score for that step
        training_data_save = np.array(training_data)
        np.save('saved.npy', training_data_save)  # saves the training data

        # some stats here, to further illustrate the neural network magic!
        print('Average accepted score:', mean(accepted_scores))
        print('Median score for accepted scores:', median(accepted_scores))
        print(Counter(accepted_scores))
        return training_data


if __name__ == "__main__":

    env = gym()  # creates the enviroment
    training_data = create_training_data()

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
                print("Episode finished after {} timesteps".format(t+1))
                break


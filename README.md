# CartPole - Stabilization of an inverted pendulum using neural networks
The inverted pendulum mounted on a cart moving in one dimension (also know as a CartPole) has long been seen as a splendid classroom example for an unstable, higly nonlinear system. Which means the system can't be modeled succesfully using only linear equations. 
Here machine learning, more specifically an artificial neural network is implemented.

This repository is the basis of a bachelor's degree thesis in mathematics at the undergraduate level, with a focus on optimization and system theory at the SCI school at the Royal Institute of Technology, by the graduate students Gustaf Holte and Måns Rasmussen.

## Artificial neural networks
Artificial nerual network, ANN, is a computer system that is primarily inspired by the biological neural network found in the human body. The network uses many different algorithms that work together to process and solve complicated problems. Such a network learns from what the user thinks is good and what its future use should be. For example, it may be "reinforced" or "supervised" learning.

In reinforced learning, one applies a scoring system to achieve a good model. The idea is that you want to get the network to understand that such a high score as possible is best and based on it, strive for high points and avoid low points. How such a scoring system can be designed can look different, but the most common one is to specify a point limit that the network must achieve in order for the model to be approved. After that, the network has to run through a number of iterations and the models that achieve the requirements are saved for further training or use.

If you want to learn more you can read at the [Neural network wikipedia page](https://en.wikipedia.org/wiki/Neural_network).

### Documentation
The gym_code is based on openAI's gym package for the enviroment (which you can find below) and is somewhat re-written to have it start from the stable position pointing straight down. 
A point system is also implemented to further seperate the good runs from the bad.
The training data is saved to hdf5 files. 
The state variables for the CartPole was the following: 
\begin{equation}
    \begin{bmatrix}
        x & \dot{x} & \theta & \dot{\theta} & L & m & M
    \end{bmatrix}
\end{equation}

The actions alowed was choosen from the force vector: 
\begin{equation} \label{forces}
    F =
    \begin{bmatrix}
        -300 & -200 & -100 & 0 & 100 & 200 & 300
    \end{bmatrix}
    N.
\end{equation} 
#### Gym's repository
Cartpole simulation
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py


_Version 12.0_
_(Version numbers adhere to [semantic versioning](https://semver.org/))_

*By Måns Rasmussen and Gustaf Holte*

This work is licensed under a [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/)

Last eddited: 16 April 2019

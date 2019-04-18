"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random as rnd

class CartPoleEnv(gym.Env):

    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(6)
        Num	Observation                 Min         Max
        0	Cart Position                7           7
        1	Cart Velocity              -Inf         Inf
        2	Pole Angle                  -90°         90°
        3	Pole Velocity At Tip        -Inf         Inf
        4   Cartpole total mass
        5   Pole length

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination: (ändrar här!)
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0  # Mass of the cart
        self.masspole = 0.1  # Mass of the pendulum (constant!)
        self.total_mass = (self.masspole + self.masscart)
        self.Length_pole = 1  # "actually half the pole's length", real length is double this
        self.polemass_length = (self.masspole * self.Length_pole)
        self.force_mag = 150.0
        self.tau = 0.02  # seconds between state updates  (time step)
        self.kinematics_integrator = 'semi-emplicit Euler' #'euler'
        self.bonus_x = 1
        self.bonus_theta = 1
        self.theta_lim_bonus = math.pi/6  # limit range for bonus points

        # Initial values of the pendulum
        self.theta_start = math.pi
        self.x_start = 0
        self.x_dot_start = 0
        self.above = False  # initialize the pendulum beneath the floor
        self.theta_dot = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 85*math.pi/180  # Not used! reset under reset(new_lim)
        self.x_threshold =  50 #np.Inf  # remove?

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max, self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max])

        #self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()  # enter number if you want to seed the random values
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def limit_check(self, theta, x):
        """Checks if the cartpole is inside the limits given"""
        if 0 <= x < self.x_threshold or - self.x_threshold < x <= 0:
            if -self.theta_lim_bonus <= theta < 0 or \
                    0 < theta <= self.theta_lim_bonus or \
                    2 * math.pi - self.theta_lim_bonus <= theta < 2 * math.pi or \
                    2 * math.pi < theta <= 2 * math.pi + self.theta_lim_bonus:
                self.bonus_theta = 3
            else:
                self.bonus_theta = 1

            if 0 <= x < self.x_threshold*2/4 or - self.x_threshold*2/4 < x <= 0:
                """Gives bonus points if inside this x range"""
                self.bonus_x = 3
            else:
                self.bonus_x = 1


            """Checks if the cartpole is inside the tracks limits"""
            if ((2*math.pi - self.theta_threshold_radians) <= theta <= 2*math.pi) or (2*math.pi <= theta <= 2*math.pi + self.theta_threshold_radians):
                """Checks if the pendulum is above ground (swingup clockwise), if so, set self.above = True"""
                self.above = True
            elif (- self.theta_threshold_radians <= theta <= 0) or (0 <= theta <= self.theta_threshold_radians):
                """Checks if the pendulum is above ground (swingup anti-clockwise), if so, set self.above = True"""
                self.above = True

            if self.above is True:
                if self.theta_threshold_radians == math.pi/2:
                    if (3/2*math.pi >= theta >= math.pi/2) or (theta > 5/2*math.pi) or (theta < - math.pi/2):
                        # pendeln har passerat marken
                        # avbryt!
                        return True
                if ((2 * math.pi + self.theta_threshold_radians) <= theta) \
                        or (3/2*math.pi <= theta <= 2 * math.pi - self.theta_threshold_radians) \
                        or (self.theta_threshold_radians <= theta <= math.pi/2) \
                        or (theta <= - self.theta_threshold_radians):
                    return True
                else:
                    return False
        else:
            return True

    def step(self, action, see_training=False):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # unpacks the initial state values
        if see_training is not False:
            state = see_training
        else:
            state = self.state
        x, x_dot, theta, theta_dot, length, polemass, masscart = state
        polemass_length = polemass*length
        total_mass = masscart + polemass

        # Set the force depending on the action
        #force = self.force if action == 1 else -self.force
        force = action

        # Calculates the force (non-linear equations)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    length * (4.0 / 3.0 - polemass * costheta * costheta / total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        theta_old = theta  # saves the old theta
        # Solves the non-linear equations
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot, length, polemass, masscart)  # updates the state values

        # Checks if the pendulum is positioned over the cart
        done = self.limit_check(theta, x)

        # Point giver
        if done is False and self.above is True:
            reward = 1.0*(self.bonus_x*self.bonus_theta)
        else:
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self, theta_lim):
        self.theta_threshold_radians = theta_lim*math.pi/180
        self.Mass_pole = self.np_random.uniform(low=0.1, high=1)
        self.Length_pole = self.np_random.uniform(low=1, high=2)
        self.masscart = self.np_random.uniform(low=1, high=2)
        x = self.x_start
        x_dot = self.x_dot_start
        theta = self.theta_start
        theta_dot = self.theta_dot  #self.np_random.uniform(low=7, high=8)*rnd.randrange(-1,2, 2)  # 7-8 is enough force to be able to swing the pendelum
        length = self.Length_pole
        polemass = self.Mass_pole
        masscart = self.masscart
        self.state = (x, x_dot, theta, theta_dot, length, polemass, masscart)
        self.above = False
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 1000
        screen_height = 600
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (self.x_threshold/6 * self.Length_pole)  # self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

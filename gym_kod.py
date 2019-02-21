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
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -Inf            Inf
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -360°           360°
        3	Pole Velocity At Tip      -Inf            Inf

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
        self.length = 0.5  # "actually half the pole's length", real length is double this
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 150.0
        self.tau = 0.02  # seconds between state updates  (time step)
        self.kinematics_integrator = 'euler'

       # Limit angles for the pendulum
        self.pole_limit_angle = 45*math.pi/180  # Limit angle for the pendulum

        # Initial values of the pendulum
        self.theta_start = math.pi
        self.x_start = 0
        self.x_dot_start = 0
        self.above = False  # initialize the pendulum beneath the floor
        self.theta_dot = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = self.pole_limit_angle  # 12 * 2 * math.pi / 360
        self.x_threshold = 5 #np.Inf  # remove?

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

        if 0 <= x < 2*self.x_threshold or - 2*self.x_threshold < x <= 0:
            """Checks if the cartpole is inside the tracks limits"""
            if (2*math.pi - self.pole_limit_angle) <= theta <= 2*math.pi or 2*math.pi <= theta <= 2*math.pi + self.pole_limit_angle:
                """Checks if the pendulum is above ground (swingup clockwise), if so, set self.above = True"""
                self.above = True
            elif (- self.pole_limit_angle <= theta <= 0) or (0 <= theta <= self.pole_limit_angle):
                """Checks if the pendulum is above ground (swingup anti-clockwise), if so, set self.above = True"""
                self.above = True

            if self.above is True:
                if ((2 * math.pi + self.pole_limit_angle) < theta) or (3/2*math.pi < theta < 2 * math.pi - self.pole_limit_angle) \
                    or (self.pole_limit_angle < theta < math.pi/2) or (theta < - self.pole_limit_angle):
                    # pendeln har passerat marken
                    # avbryt!
                    return True
                else:
                    return False
        else:
            return True

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # unpacks the initial state values
        state = self.state
        x, x_dot, theta, theta_dot, length, polemass, force = state
        polemass_length = polemass*length
        total_mass = self.masscart + polemass

        # Set the force depending on the action
        #force = self.force if action == 1 else -self.force
        if action is not None:
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
        action = force
        self.state = (x, x_dot, theta, theta_dot, length, polemass, action)  # updates the state values

        # Checks if the pendulum is positioned over the cart
        done = self.limit_check(theta, x)

        # Point giver
        if done is False and self.above is True:
            #print("you get one point!")
            reward = 1.0
        else:
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=1, high=10000, size=(4,))  # creates a random uniform array = [x, x_dot, theta, theta_dot]
        self.M = self.np_random.uniform(low=1, high=2)
        self.L = self.np_random.uniform(low=1, high=2)
        force = rnd.randrange(start=-150, stop=150, step=10)#self.force_mag  #rnd.randrange(50,300, 10)##
        x = self.x_start
        x_dot = self.x_dot_start
        theta = self.theta_start
        theta_dot = self.theta_dot #self.np_random.uniform(low=7, high=8)*rnd.randrange(-1,2, 2)  # 7-8 is enough force to be able to swing the pendelum
        length = self.L
        polemass = self.M #self.np_random.uniform(low=0.1, high=1)
        self.state = (x, x_dot, theta, theta_dot, length, polemass, force)
        self.above = False
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
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
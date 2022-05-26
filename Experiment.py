import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from Agents import MonteCarloG
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SkipFrame(gym.Wrapper):
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.n_frames):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info


def main():
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(SkipFrame(env, n_frames=4), keep_dim=True)
    # env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, n_frames=4)), shape=84), num_stack=4)

    n_time_steps = 1000
    gamma = 0.99
    learning_rate = 0.001

    pi = MonteCarloG(env.observation_space.shape, env.action_space.n, learning_rate, gamma)

    for episodes in range(10):
        s = env.reset()
        s_list = []
        r_list = []
        a_list = []
        done = False
        for i in range(100):

            a, p = pi.select_action(s)

            # check p to see how it changes
            if i==0:
                print(p)

            s_next, r, done, info = env.step(a)
            # env.render()
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)

            if done or i == 99:
                print('done')
                pi.update(r_list,a_list,s_list)
                print(np.sum(r_list))
                break

            else:
                s = s_next


if __name__ == '__main__':
    main()

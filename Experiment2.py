import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from Agents import MonteCarloG  
import gym
import matplotlib.pyplot as plt
from Helper import LearningCurvePlot, smooth, ComparisonPlot

def run_repetitions(rl_method, env, n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps):
    learning_curve = np.zeros(shape=(n_repetitions, n_episodes))
    for rep in range(n_repetitions):
        print(f'Repetition {rep} ' + rl_method)
        if rl_method == 'MonteCarloG':
            pi = MonteCarloG(env.observation_space.shape, env.action_space.n, learning_rate, gamma)
        for e in range(n_episodes):
            s = env.reset()
            s_list = []
            r_list = []
            a_list = []
            done = False
            for i in range(max_steps):
                a, p = pi.select_action(s)
                s_next, r, done, info = env.step(a)
                # env.render()
                s_list.append(s)
                a_list.append(a)
                r_list.append(r)

                if done or i == max_steps-1:
                    #print(f'done {e}')
                    pi.update(r_list,a_list,s_list)
                    learning_curve[rep, e] = np.sum(r_list)
                    break

                else:
                    s = s_next
    learning_curve = np.mean(learning_curve, axis=0)
    learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve
                



def main():
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym.make('CartPole-v1')
    # env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, n_frames=4)), shape=84), num_stack=4)

    gamma = 0.99
    learning_rate = 0.001
    max_steps = 50
    n_repetitions = 5
    n_episodes = 100
    smoothing_window = 11   
        
    Plot = LearningCurvePlot('Graph_Test')
    r = run_repetitions('MonteCarloG', env, n_repetitions, n_episodes, smoothing_window, learning_rate, gamma, max_steps)
    Plot.add_curve(r, label='')
    Plot.save('Graph_Test')
    
if __name__ == '__main__':
    main()

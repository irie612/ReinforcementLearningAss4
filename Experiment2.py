import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from Agents import MonteCarloG  
import gym
import matplotlib.pyplot as plt


def main():
    # env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym.make('CartPole-v1')
    # env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, n_frames=4)), shape=84), num_stack=4)

    gamma = 0.99
    learning_rate = 0.001

    pi = MonteCarloG(env.observation_space.shape, env.action_space.n, learning_rate, gamma)

    rewards = []
    
    E = 100
    
    for episode in range(E):
        s = env.reset()
        s_list = []
        r_list = []
        a_list = []
        done = False
        while True:
            
            
            a, p = pi.select_action(s)
        
            # check p to see how it changes
            # if i==0:
            #env.render()
            s_next, r, done, info = env.step(a)
            # env.render()
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)

            if done:
                #print(p)
                print(f'done {episode}')
                #print(p)
                pi.update(r_list,a_list,s_list)
                rewards.append(np.sum(r_list))
                break

            else:
                s = s_next
            
    plt.plot([i for i in range(E)], rewards)
    plt.show()

if __name__ == '__main__':
    main()

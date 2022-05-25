import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from Agents import MonteCarloG

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    
    n_time_steps = 1000
    gamma = 0.99
    learning_rate = 0.01


    pi = MonteCarloG(env.observation_space.shape , env.action_space.n, learning_rate, gamma)

    for episodes in range(10):
        s = env.reset()
        s_list = []
        r_list = []
        a_list = []     
        done = False
        for i in range(1000):
            
            a, p = pi.select_action(s)
            
            # check p to see how it changes
            if i==0:
                print(p)
        
            s_next, r, done, info = env.step(a)
            #env.render()
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
    
    
            if done or i==999:
                print('done')
                pi.update(r_list,a_list,s_list)
                print(np.sum(r_list))
                break
                
            else:
                s = s_next

if __name__ == '__main__':
    main()

from multiprocessing import Pool, cpu_count
import numpy as np
from Agents import MonteCarloG
import gym
from Helper import smooth, plot_colormap, plot_reward_graph
from tqdm import tqdm


def run_episodes(agent, env_name, n_episodes, alpha, gamma, max_steps):
    rewards_over_episodes = np.zeros(shape=n_episodes)
    env = gym.make(env_name)
    pi = agent(env.observation_space.shape, env.action_space.n, alpha, gamma)
    for e in range(n_episodes):
        s = env.reset()
        s_list = []
        r_list = []
        a_list = []
        for i in range(max_steps):
            a, p = pi.select_action(s)
            s_next, r, done, info = env.step(a)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            if done or i == max_steps - 1:
                pi.update(r_list, a_list, s_list)
                rewards_over_episodes[e] = np.sum(r_list)
                break
            else:
                s = s_next
    return rewards_over_episodes


def run_repetitions(agent, env_name, n_repetitions, n_episodes, smoothing_window, alpha, gamma, max_steps):
    learning_curve = np.zeros(shape=(n_repetitions, n_episodes))
    for rep in range(n_repetitions):
        print(f'Repetition {rep} ' + agent.__name__)
        learning_curve[rep] = run_episodes(agent, env_name, n_episodes, alpha, gamma, max_steps)
    learning_curve = np.mean(learning_curve, axis=0)
    learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve


def run_repetitions_parallel(agent, env_name, n_reps, n_eps, smoothing_win, alpha, gamma, max_steps):
    async_results = []
    pool = Pool(processes=cpu_count())
    for i in range(n_reps):
        async_results.append(pool.apply_async(run_episodes, args=(agent, env_name, n_eps, alpha, gamma, max_steps)))
    pool.close()
    pool.join()
    learning_curve = np.mean([r.get() for r in async_results], axis=0)
    learning_curve = smooth(learning_curve, smoothing_win)
    return learning_curve


def run_experiment(alphas, gammas, agent, env_name, n_reps, n_eps, smoothing_win, max_steps, parallel=True):
    means = np.zeros(shape=(len(alphas), len(gammas), n_eps))
    for i, alpha, in enumerate(alphas):
        for j, gamma in enumerate(tqdm(gammas, desc=f"{agent.__name__} in {env_name} for {alpha=} over {gammas=}")):
            if parallel:
                means[i, j] = run_repetitions_parallel(agent, env_name, n_reps, n_eps, smoothing_win, alpha, gamma, max_steps)
                np.save(f"arrays/{alpha}_{gamma}", means[i, j])
            else:
                means[i, j] = run_repetitions(agent, env_name, n_reps, n_eps, smoothing_win, alpha, gamma, max_steps)
    return means


def main():
    env_name = 'CartPole-v1'
    agent = MonteCarloG
    gammas = [0.9, 0.95, 0.99, 1]
    alphas = [0.00001, 0.0001, 0.001, 0.01]
    max_steps = 50
    n_repetitions = 5
    n_episodes = 200
    smoothing_window = 11

    means = run_experiment(alphas, gammas, agent, env_name, n_repetitions, n_episodes, smoothing_window, max_steps)
    np.save("arrays/means", means)


if __name__ == '__main__':
    main()

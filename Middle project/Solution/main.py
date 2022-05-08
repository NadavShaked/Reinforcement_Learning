from enum import Enum
from typing import Tuple, List, Union, Optional, NamedTuple
import numpy as np
import random
import matplotlib.pyplot as plt

from maze import MazeEnvSample15x15, MazeEnvSample30x30, MazeEnv


class Algorithm(str, Enum):
    MonteCarlo = 'MonteCarlo'
    QLearning = 'QLearning'


class ValueMethod(str, Enum):  # TODO: implement
    FirstVisit = 'FirstVisit'
    EveryVisit = 'EveryVisit'


class EvaluationMethod(str, Enum):
    State = 'State'
    ActionState = 'ActionState'


class PolicyArgs(NamedTuple):
    alpha: Optional[float]
    gamma: Optional[float]
    epsilon: Optional[float]
    stochastic_probability: float
    is_epsilon_decrease: Optional[bool]


StateType = Tuple[int, int]
StateActionType = Tuple[StateType, str]
MAX_STEP = 2500
NUM_OF_EPISODES = 500
K = 25


def state_as_tuple(state):
    return int(state[0]), int(state[1])


def get_action_highest_value(env, state_values):
    max_value = -np.inf
    opt_action = None

    for action in env.ACTION:
        cur_state = state_as_tuple(env.state)
        direction = env.maze_view.maze.COMPASS[action]
        next_state = cur_state[0] + direction[0], cur_state[1] + direction[1]

        if not env.maze_view.maze.is_within_bound(next_state[0], next_state[1]):
            next_state = cur_state

        next_state_value = state_values[next_state[0], next_state[1]]
        if next_state_value > max_value:
            max_value = next_state_value
            opt_action = action

    return opt_action


def calc_action(env, cur_state, evaluation_method, epsilon, state_policy):
    """ Calculate action according to policy """
    action = None
    if evaluation_method == EvaluationMethod.State:
        action = get_action_highest_value(env, state_policy)
    elif evaluation_method == EvaluationMethod.ActionState:
        action = state_policy[cur_state[0]][cur_state[1]]

    if epsilon:
        action = implement_action_probability(action, env.ACTION, 1 - epsilon)

    return action


def improve_monte_carlo_policy(state_list, rewards, returns, state_policy, state_action_values, evaluation_method,
                               args):
    G = 0
    for i in reversed(range(len(state_list))):
        G = args.gamma * G + rewards[i]
        if state_list[i] not in state_list[:i - 1]:
            if returns.get(state_list[i]):
                returns[state_list[i]].append(G)
            else:
                returns[state_list[i]] = [G]

            if evaluation_method == EvaluationMethod.State:
                state_policy[state_list[i][0]][state_list[i][1]] = np.average(returns[state_list[i]])
            elif evaluation_method == EvaluationMethod.ActionState:
                state_action_values[state_list[i][0][0]][state_list[i][0][1]][
                    env.ACTION.index(state_list[i][1])] = np.average(returns[state_list[i]])
                state_policy[state_list[i][0][0]][state_list[i][0][1]] = env.ACTION[
                    np.argmax(state_action_values[state_list[i][0][0]][state_list[i][0][1]])]


def improve_q_learning_policy(env, cur_state, cur_action, reward, state_actions, state_policy, args):
    next_state = state_as_tuple(env.state)
    next_action = calc_action(env, next_state, EvaluationMethod.ActionState, args.epsilon, state_policy)

    q_s_a = state_actions[cur_state[0]][cur_state[1]][env.ACTION.index(cur_action)]
    q_s_a += args.alpha * (reward + args.gamma * state_actions[next_state[0]][next_state[1]][
        env.ACTION.index(next_action)] - q_s_a)
    state_actions[cur_state[0]][cur_state[1]][env.ACTION.index(cur_action)] = q_s_a
    state_policy[cur_state[0]][cur_state[1]] = env.ACTION[np.argmax(state_actions[cur_state[0]][cur_state[1]])]


def implement_action_probability(action, action_list, probability):
    if len(action_list) <= 1:
        return action

    opt_action_index = action_list.index(action)
    non_opt_action_probability = (1 - probability) / (len(action_list) - 1)
    probabilities = np.full(len(action_list), non_opt_action_probability)
    probabilities[opt_action_index] = probability

    return random.choices(action_list, probabilities)[0]


def initialize_values(env, policy, evaluation_method):
    state_policy = None
    state_action_values = None
    if evaluation_method == EvaluationMethod.State:
        state_policy = -np.random.random_sample(size=(env.maze_size[1], env.maze_size[0]))
    elif evaluation_method == EvaluationMethod.ActionState:
        state_policy = np.random.choice(env.ACTION, size=(env.maze_size[1], env.maze_size[0]))
        state_action_values = -np.random.random_sample(size=(env.maze_size[1], env.maze_size[0], len(env.ACTION)))

        if policy.QLearning:
            for i in range(len(env.ACTION)):  # init terminal state to 0
                state_action_values[env.maze_size[1] - 1][env.maze_size[0] - 1][i] = 0

    return state_action_values, state_policy


def run_policy(env: MazeEnv, algorithm: Algorithm, evaluation_method: EvaluationMethod,
               args: PolicyArgs, exploration: bool = False, overwrite_rewards={}):
    episode_steps_counts = []
    # Initialize values
    returns = {}
    state_action_values, state_policy = initialize_values(env, algorithm, evaluation_method)

    # Run episodes
    for episode_number in range(NUM_OF_EPISODES):
        env.reset()
        if exploration and episode_number < NUM_OF_EPISODES - 10 and episode_number % K != 0:  # Always start last 10 episodes from starting point
            env.state = np.random.randint(0, env.maze_view.maze_size, size=2)

        done = False
        state_list: List[Union[StateType, StateActionType]] = []
        rewards = []
        cur_epsilon = args.epsilon * (1 - episode_number / NUM_OF_EPISODES) \
            if args.epsilon and args.is_epsilon_decrease else args.epsilon
        step_cnt = 0

        # generate an episode
        while not done:
            if step_cnt == MAX_STEP:
                break

            step_cnt += 1
            state = state_as_tuple(env.state)
            action = calc_action(env, state, evaluation_method, cur_epsilon, state_policy)
            action = implement_action_probability(action, env.ACTION, args.stochastic_probability)

            # Take step
            observation, reward, done, info = env.step(action)

            if overwrite_rewards.get(state_as_tuple(observation)):
                reward = overwrite_rewards.get(state_as_tuple(observation))

            if algorithm == Algorithm.MonteCarlo:
                # Record step
                if evaluation_method == EvaluationMethod.State:
                    state_list.append(state)
                elif evaluation_method == EvaluationMethod.ActionState:
                    state_list.append((state, action))
                rewards.append(reward)
            elif algorithm == Algorithm.QLearning:
                # improve policy
                improve_q_learning_policy(env, state, action, reward, state_action_values, state_policy, args)

        if algorithm == Algorithm.MonteCarlo:
            # improve policy
            improve_monte_carlo_policy(state_list, rewards, returns, state_policy, state_action_values,
                                       evaluation_method, args)

        if episode_number % K == 0:
            print(f"Generated an episode {episode_number} -- took {step_cnt} steps")
            episode_steps_counts.append(step_cnt)

    return episode_steps_counts


if __name__ == '__main__':
    env = MazeEnvSample15x15()
    overwrite_rewards = {}

    if isinstance(env, MazeEnvSample30x30):
        overwrite_rewards = {
            (7, 23): 0,
            (22, 6): 0
        }

    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for epsilon in [None, 0.2, 0.25, 0.3]:
            for is_epsilon_decrease in [True, False]:
                for exploration in [True, False]:
                    alpha = alpha
                    gamma = 1.0
                    # epsilon = 0.2
                    # is_epsilon_decrease = False

                    policy_args = PolicyArgs(
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                        stochastic_probability=0.9,
                        is_epsilon_decrease=is_epsilon_decrease
                    )

                    algorithm = Algorithm.QLearning
                    evaluation_method = EvaluationMethod.ActionState
                    # exploration = True

                    episode_steps_counts = run_policy(env=env,
                                                      algorithm=algorithm,
                                                      evaluation_method=evaluation_method,
                                                      exploration=exploration,
                                                      args=policy_args,
                                                      overwrite_rewards=overwrite_rewards)


                    episodes = [i * K for i in range(len(episode_steps_counts))]
                    plt.plot(episodes, episode_steps_counts)

                    plt.xlabel("Episode number")
                    plt.ylabel("Number of steps")

                    plt.savefig(f'{algorithm}_{evaluation_method}/{algorithm}_{evaluation_method}_{"exploration" if exploration else "nonExploration"}_{alpha}_{gamma}_{epsilon}_{"epsilonDecrease" if is_epsilon_decrease else "epsilonConstant"}.png')
                    plt.show()

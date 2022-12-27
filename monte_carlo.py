#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=8000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.27, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_decay", default=1, type=float, help="Exploration factor decay.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float32)
    C = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.int32)

    epsilon = args.epsilon

    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
              env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of `args.epsilon`, use a random action,
            # otherwise choose an action with maximum `Q[state, action]`.

            if rng.random() <= epsilon:
                action = rng.integers(0, env.action_space.n)
            else:
                action = np.argmax(Q[state, :])

            # Perform the action.
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0
        states.reverse(), actions.reverse(), rewards.reverse()
        for s, a, r in zip(states, actions, rewards):
            G += r  # G = args.gamma * G + r
            C[s, a] += 1
            Q[s, a] += (1 / C[s, a]) * (G - Q[s, a])

        epsilon *= args.epsilon_decay

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)

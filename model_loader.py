#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=15, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1800, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")

class Agent:
    def __init__(self, policy_model):
        self._model = policy_model

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    if args.recodex:
        # We run in ReCodEx, perform evaluation with a trained model
        policy_model = tf.keras.models.load_model('policy_model.h5')
        agent = Agent(policy_model)
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                action = np.argmax(np.squeeze(agent.predict(np.expand_dims(state, axis=0))))
                state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
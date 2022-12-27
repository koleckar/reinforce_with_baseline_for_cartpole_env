#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=1024, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")


class Agent:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with a single output without an activation).
        # (Alternatively, this baseline computation can be grouped together
        # with the policy computation in a single `tf.keras.Model`.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.

        inputs = tf.keras.layers.Input(env.observation_space.shape, name='input_layer')

        flatten = tf.keras.layers.Flatten(name='flatten_layer')(inputs)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu', name='hidden_layer')(flatten)
        output = tf.keras.layers.Dense(env.action_space.n, activation='sigmoid', name='output_layer')(hidden)
        self._model = tf.keras.Model(inputs=inputs, outputs=output, name='REINFORCE_model')
        self._model_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._model_loss = tf.keras.losses.CategoricalCrossentropy()

        hidden_2 = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu', name='baseline_dense')(flatten)
        output_2 = tf.keras.layers.Dense(1, activation=None, name='baseline_out')(hidden_2)
        self._baseline_model = tf.keras.Model(inputs=inputs, outputs=output_2, name='Baseline_model')
        self._baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._baseline_loss = tf.losses.MeanSquaredError()

    # Define a training method.
    #
    # Note that we need to use @tf.function and manual `tf.GradientTape`
    # for efficiency (using `fit` or `train_on_batch` on extremely small
    # batches has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Perform training, using the loss from the REINFORCE with baseline algorithm.
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as the advantage estimate

        with tf.GradientTape() as tape:
            predicted_baseline = self._baseline_model(states, training=True)
            baseline_loss = self._baseline_loss(y_true=returns, y_pred=predicted_baseline)
        gradients = tape.gradient(baseline_loss, self._baseline_model.trainable_variables)
        self._baseline_optimizer.apply_gradients(zip(gradients, self._baseline_model.trainable_variables))

        with tf.GradientTape() as tape:
            predicted_actions_probabilities = self._model(states, training=True)
            gold_actions = tf.one_hot(indices=actions, depth=2)
            loss = self._model_loss(y_true=gold_actions, y_pred=predicted_actions_probabilities,
                                    sample_weight=returns - tf.squeeze(tf.stop_gradient(predicted_baseline)))
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._model_optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                p = agent.predict(np.expand_dims(state, axis=0))  # need to expand dims as batch of 1
                p /= np.sum(p)  # np issue
                action = rng.choice(a=env.action_space.n, p=np.squeeze(p))

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
            # TODO(reinforce): Compute returns from the received rewards
            G = []
            T = len(rewards)
            for t in range(T):
                discounted_rewards = np.sum(args.gamma ** np.arange(T - t) * rewards[t: T])
                G.append(discounted_rewards)

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += G

        # TODO(reinforce): Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO(reinforce): Choose greedy action
            action = np.argmax(np.squeeze(agent.predict(np.expand_dims(state, axis=0))))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed)

    main(env, args)

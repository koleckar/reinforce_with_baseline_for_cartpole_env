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
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1600, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=150, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")


def conv_block(layer_input, filters, name=''):
    return tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', name=name + '_pool')(
        tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation='relu',
                               name=name + '_conv')(layer_input))


class Agent:

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        super().__init__()

        inputs = tf.keras.layers.Input(env.observation_space.shape, name='input_layer')

        c1 = conv_block(inputs, filters=16, name='c1')
        c2 = conv_block(c1, filters=32, name='c2')
        c3 = conv_block(c2, filters=64, name='c3')
        c4 = conv_block(c3, filters=128, name='c4')
        flatten = tf.keras.layers.Flatten()(c4)
        hidden = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu')(flatten)
        dropout = tf.keras.layers.Dropout(args.dropout)(hidden)
        hidden2 = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu')(dropout)
        dropout2 = tf.keras.layers.Dropout(args.dropout)(hidden2)
        output = tf.keras.layers.Dense(env.action_space.n, activation='sigmoid', name='output_layer')(dropout2)
        self._model = tf.keras.Model(inputs=inputs, outputs=output, name='REINFORCE_model')
        self._model_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._model_loss = tf.keras.losses.CategoricalCrossentropy()
        self._model.summary()

        c1_b = conv_block(inputs, filters=16, name='c1')
        c2_b = conv_block(c1_b, filters=32, name='c2')
        c3_b = conv_block(c2_b, filters=64, name='c3')
        c4_b = conv_block(c3_b, filters=128, name='c4')
        flatten_b = tf.keras.layers.Flatten()(c4_b)
        hidden_b = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu')(flatten_b)
        dropout_b = tf.keras.layers.Dropout(args.dropout)(hidden_b)
        hidden2_b = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu')(dropout_b)
        dropout2_b = tf.keras.layers.Dropout(args.dropout)(hidden2_b)
        output_b = tf.keras.layers.Dense(1, activation=None, name='baseline_out')(dropout2_b)
        self._baseline_model = tf.keras.Model(inputs=inputs, outputs=output_b, name='Baseline_model')
        self._baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._baseline_loss = tf.losses.MeanSquaredError()
        self._baseline_model.summary()

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:

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
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    # Construct the agent
    agent = Agent(env, args)

    # TODO: Perform training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                p = agent.predict(np.expand_dims(state, axis=0))  # need to expand dims as batch of 1
                p /= np.sum(p)  # np issue
                action = rng.choice(a=env.action_space.n, p=np.squeeze(p))

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
            # Compute returns from the received rewards
            G = []
            T = len(rewards)
            for t in range(T):
                discounted_rewards = np.sum(args.gamma ** np.arange(T - t) * rewards[t: T])
                G.append(discounted_rewards)

            # Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += G

        # Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)

    agent._model.save('policy_model.h5')

    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose an action
            action = np.argmax(np.squeeze(agent.predict(np.expand_dims(state, axis=0))))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)

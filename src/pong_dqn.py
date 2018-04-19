import sys
import os
import logging.config
from time import time, sleep


import gym
import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

from collections import deque

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'root': {
            'handlers': ['default'],
            'level': 'INFO'
        },
        'PongDQN': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    }
}


logging.config.dictConfig(DEFAULT_LOGGING)

logger = logging.getLogger('PongDQN')


class PConf:
    PROJECT_ROOT_DIR = os.environ['DQL_ROOT_DIR']
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, 'output')

    TOTAL_TRAINING_STEPS = 4000000  # total number of training steps
    TRAINING_START = 10000  # start training after 10,000 game iterations
    TRAINING_INTERVAL = 4  # run a training step every 4 game iterations
    SAVE_INTERVAL = 100000  # save the model every 1,000 training steps
    COPY_INTERVAL = 10000  # copy online DQN to target DQN every 10,000 training steps

    MV = 'MV01R00'

    OV = 'OV01R00'
    LEARNING_RATE = 0.005

    REPLAY_MEMORY_SIZE = 500000
    REPLAY_BATCH_SIZE = 50

    GAMMA = 0.85 # discount rate

    EPS_MIN = 0.1
    EPS_MAX = 1.0
    EPS_DECAY_STEPS = 2000000

    SAVE_MODEL = 'TR001'


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)

    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


class DQN:
    def __init__(self, observation_space, action_space):
        self.learning_rate = PConf.LEARNING_RATE

        self.replay_memory_size = PConf.REPLAY_MEMORY_SIZE
        self.replay_batch_size = PConf.REPLAY_BATCH_SIZE

        self.gamma = PConf.GAMMA

        self.eps_min = PConf.EPS_MIN
        self.eps_max = PConf.EPS_MAX
        self.eps_decay_steps = PConf.EPS_DECAY_STEPS

        self.observation_space = observation_space
        self.action_space = action_space

        self.memory = deque([], maxlen=self.replay_memory_size)

        self.online_model = self.create_model()
        self.target_model = self.create_model()


    def create_model(self):
        ip = Input(shape=(self.observation_space.shape[0],))
        print(ip.shape)
        x = Dense(200, activation='relu')(ip)
        print(x.shape)
        x = Dense(200, activation='relu')(x)
        print(x.shape)
        x = Dense(200, activation='relu')(x)
        print(x.shape)
        q = Dense(self.action_space.n)(x)
        print(q.shape)

        model = Model(inputs=[ip], outputs=q)

        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        return model


    def act(self, state, step):
        eps = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)

        if np.random.rand() < eps:
            return self.action_space.sample()  # random action
        else:
            state = state.reshape(1, 128)  # preprocess later ..

            return np.argmax(self.online_model.predict(state)[0])


    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])


    def sample_memories(self, batch_size):
        indices = np.random.permutation(len(self.memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, done

        for idx in indices:
            experience = self.memory[idx]

            for col, value in zip(cols, experience):
                col.append(value)

        cols = [np.array(col) for col in cols]

        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


    def replay_and_fit(self):
        if len(self.memory) >= self.replay_batch_size:
            state, action, reward, next_state, done = self.sample_memories(self.replay_batch_size)

            target_q_values = self.target_model.predict(state)

            Q_future = np.max(self.target_model.predict(next_state), axis=1)

            target_q_values[:, action] = reward + (1.0 - done) * Q_future * self.gamma

            history = self.online_model.fit(state, target_q_values, epochs=1, verbose=0)

            return np.mean(history.history['loss'])


    def copy_online_to_target(self):
        online_weights = self.online_model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = online_weights[i] #* self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)


    def save_model(self):
        self.online_model.save(os.path.join(PConf.OUTPUT_DIR, PConf.SAVE_MODEL + '.h5'))

    @staticmethod
    def load_model():
        model = load_model(os.path.join(PConf.OUTPUT_DIR, PConf.SAVE_MODEL + '.h5'))

        return model


def execute_training():
    env = gym.make('Pong-ramNoFrameskip-v4')

    done = True  # env needs to be reset
    state = None
    step = 0
    loss = None
    ex_duration = 0
    w_exchanges = []
    l_exchanges = []

    dqn_agent = DQN(env.observation_space, env.action_space)

    while step < PConf.TOTAL_TRAINING_STEPS:
        if done:
            # game over, start again
            state = env.reset()

        # online DQN evaluates what to do
        action = dqn_agent.act(state, step)

        # online DQN plays
        next_state, reward, done, info = env.step(action)

        if (reward == 0):
            ex_duration += 1
        elif (reward > 0):
            w_exchanges.append(ex_duration)
            ex_duration = 0
        elif (reward < 0):
            l_exchanges.append(ex_duration)
            ex_duration = 0

        dqn_agent.remember(state, action, reward, next_state, done)

        if step >= PConf.TRAINING_START:
            if step % PConf.TRAINING_INTERVAL == 0:
                # only train after warmup period and at regular intervals
                loss = dqn_agent.replay_and_fit()

            if step % PConf.COPY_INTERVAL == 0:
                dqn_agent.copy_online_to_target()

            if step % PConf.SAVE_INTERVAL == 0:
                dqn_agent.save_model()

            if ((len(l_exchanges) + len(w_exchanges)) % 10 == 0) and (len(l_exchanges) > 0 or len(w_exchanges) > 0):
                logger.info("Training result at %s %% progress: %s, %s, %s, %s, %s",
                            step * 100.0 / PConf.TOTAL_TRAINING_STEPS, loss, len(l_exchanges), np.mean(l_exchanges),
                            len(w_exchanges), np.mean(w_exchanges))

                l_exchanges.clear()
                w_exchanges.clear()

        step += 1



def execute_test(model, render):
    env = gym.make('Pong-ramNoFrameskip-v4')

    state = env.reset()
    done = False
    step = 0

    while step < 10000 and not done:
        sleep(0.01)

        if render:
            env.render()

        state = state.reshape(1, 128)  # preprocess later ..
        action = np.argmax(model.predict(state)[0])

        state, _, done, _ = env.step(action)


def main():
    overall = time()

    logger.info("Main script started ...")

    training = False
    test = False

    render = False

    for arg in sys.argv[1:]:
        if arg == 'training':
            training = True
        elif arg == 'test':
            test = True
        elif arg == 'render':
            render = True

    if not test and not training:
        training = True

    if training:
        execute_training()

    if test:
        model = DQN.load_model()

        execute_test(model, render)

    logger.info("Main script finished in %s.", time_it(overall, time()))


if __name__ == "__main__":
    main()
import tensorflow as tf
import random as rand
import numpy as np
from convnet import ConvNet
from buff import Buffer
from memory import Memory


class DQN:

    def __init__(self, env, params):
        self.env = env
        params.actions = env.actions()
        self.num_actions = env.actions()
        self.episodes = params.episodes
        self.steps = params.steps
        self.train_steps = params.train_steps
        self.update_freq = params.update_freq
        self.save_weights = params.save_weights
        self.history_length = params.history_length
        self.discount = params.discount
        self.eps = params.init_eps
        self.eps_delta = (params.init_eps - params.final_eps) / params.final_eps_frame
        self.replay_start_size = params.replay_start_size
        self.eps_endt = params.final_eps_frame
        self.random_starts = params.random_starts
        self.batch_size = params.batch_size
        self.ckpt_file = params.ckpt_dir+'/'+params.game

        self.global_step = tf.Variable(0, trainable=False)
        if params.lr_anneal:
            self.lr = tf.train.exponential_decay(params.lr, self.global_step, params.lr_anneal, 0.96, staircase=True)
        else:
            self.lr = params.lr

        self.buffer = Buffer(params)
        self.memory = Memory(params.size, self.batch_size)

        with tf.variable_scope("train") as self.train_scope:
            self.train_net = ConvNet(params, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = ConvNet(params, trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, params.decay_rate, 0.0, self.eps)

        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.q_target = tf.placeholder(tf.float32, [None])
        self.q_train = tf.reduce_max(tf.mul(self.train_net.y, self.actions), reduction_indices=1)
        self.diff = tf.sub(self.q_target, self.q_train)

        half = tf.constant(0.5)
        if params.clip_delta > 0:
            abs_diff = tf.abs(self.diff)
            clipped_diff = tf.clip_by_value(abs_diff, 0, 1)
            linear_part = abs_diff - clipped_diff
            quadratic_part = tf.square(clipped_diff)
            self.diff_square = tf.mul(half, tf.add(quadratic_part, linear_part))
        else:
            self.diff_square = tf.mul(half, tf.square(self.diff))

        if params.accumulator == 'sum':
            self.loss = tf.reduce_sum(self.diff_square)
        else:
            self.loss = tf.reduce_mean(self.diff_square)

        # backprop with RMS loss
        self.task = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def randomRestart(self):
        self.env.restart()
        for _ in range(self.random_starts):
            action = rand.randrange(self.num_actions)
            reward = self.env.act(action)
            state = self.env.getScreen()
            terminal = self.env.isTerminal()
            self.buffer.add(state)

            if terminal:
                self.env.restart()

    def trainEps(self, train_step):
        if train_step < self.eps_endt:
            return self.eps - train_step * self.eps_delta
        else:
            return self.eps_endt

    def observe(self, exploration_rate):
        if rand.random() < exploration_rate:
            a = rand.randrange(self.num_actions)
        else:
            x = self.buffer.getInput()
            action_values = self.train_net.y.eval( feed_dict={ self.train_net.x: x } )
            a = np.argmax(action_values)
        
        state = self.buffer.getState()
        action = np.zeros(self.num_actions)
        action[a] = 1.0
        reward = self.env.act(a)
        screen = self.env.getScreen()
        self.buffer.add(screen)
        next_state = self.buffer.getState()
        terminal = self.env.isTerminal()

        reward = np.clip(reward, -1.0, 1.0)

        self.memory.add(state, action, reward, next_state, terminal)
        
        
        return state, action, reward, next_state, terminal

    def doMinibatch(self, sess, successes, failures):
        batch = self.memory.getSample()
        state = np.array([batch[i][0] for i in range(self.batch_size)]).astype(np.float32)
        actions = np.array([batch[i][1] for i in range(self.batch_size)]).astype(np.float32)
        rewards = np.array([batch[i][2] for i in range(self.batch_size)]).astype(np.float32)
        successes += np.sum(rewards==1)
        next_state = np.array([batch[i][3] for i in range(self.batch_size)]).astype(np.float32)
        terminals = np.array([batch[i][4] for i in range(self.batch_size)]).astype(np.float32)

        failures += np.sum(terminals==1)
        q_target = self.target_net.y.eval( feed_dict={ self.target_net.x: next_state } )
        q_target_max = np.argmax(q_target, axis=1)
        q_target = rewards + ((1.0 - terminals) * (self.discount * q_target_max))

        (result, loss) = sess.run( [self.task, self.loss],
                                    feed_dict={ self.q_target: q_target,
                                                self.train_net.x: state,
                                                self.actions: actions } )

        return successes, failures, loss

    def play(self):
        self.randomRestart()
        self.env.restart()
        for i in xrange(self.episodes):
            terminal = False
            while not terminal:
                #aca cambie algo
                state, action, reward, screen, terminal = self.observe(self.eps)

    def copy_weights(self, sess):
        for key in self.train_net.weights.keys():
            t_key = 'target/' + key.split('/', 1)[1]
            sess.run(self.target_net.weights[t_key].assign(self.train_net.weights[key]))

    def save(self, saver, sess, step):
        saver.save(sess, self.ckpt_file, global_step=step)
        
    def restore(self, saver):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_file)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)



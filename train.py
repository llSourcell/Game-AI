import time
import tensorflow as tf


class Trainer:

    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env
        self.saver = tf.train.Saver()

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.agent.randomRestart()

            successes = 0
            failures = 0
            total_loss = 0

            print "start %d random plays to populate replay memory" % self.agent.replay_start_size
            for i in xrange(self.agent.replay_start_size):
                # follow random policy
                state, action, reward, next_state, terminal = self.agent.observe(1)

                if reward == 1:
                    successes += 1
                elif reward == -1:
                    failures += 1

                if (i+1) % 10000 == 0:
                    print "memory size: %d" % len(self.agent.memory)
            
            sample_success = 0
            sample_failure = 0
            print "start training..."
            start_time = time.time()
            for i in xrange(self.agent.train_steps):
                # annealing learning rate
                lr = self.agent.trainEps(i)
                state, action, reward, next_state, terminal = self.agent.observe(lr)

                if len(self.agent.memory) > self.agent.batch_size and (i+1) % self.agent.update_freq == 0:
                    sample_success, sample_failure, loss = self.agent.doMinibatch(sess, sample_success, sample_failure)
                    total_loss += loss

                if (i+1) % self.agent.steps == 0:
                    self.agent.copy_weights(sess)

                if reward == 1:
                    successes += 1
                elif reward == -1:
                    failures += 1
                
                if ((i+1) % self.agent.save_weights == 0):
                    self.agent.save(self.saver, sess, i+1)

                if ((i+1) % self.agent.batch_size == 0):
                    avg_loss = total_loss / self.agent.batch_size
                    end_time = time.time()
                    print "memory size: ", len(self.agent.memory),\
                          "\nTraining steps: ", i+1,\
                          "\nLearning rate: ", lr,\
                          "\nSuccesses: ", successes,\
                          "\nFailures: ", failures,\
                          "\nSuccesse samples: ", sample_success,\
                          "\nFailure samples: ", sample_failure,\
                          "\nAverage batch loss: ", avg_loss,\
                          "\nBatch training time: ", (end_time-start_time)/self.agent.batch_size
                    start_time = time.time()
                    total_loss = 0

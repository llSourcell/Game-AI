import os
import argparse
import random as rand
from environment import Environment
from train import Trainer
from dqn import DQN

## these are just command line arguments. The 10 line code is at the bottom -- Siraj
parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Environment')
envarg.add_argument("--game", type=str, default="SpaceInvaders-v0", help="Name of the atari game to test")
envarg.add_argument("--width", type=int, default=84, help="Screen width")
envarg.add_argument("--height", type=int, default=84, help="Screen height")

memarg = parser.add_argument_group('Memory')
memarg.add_argument("--size", type=int, default=100000, help="Memory size.")
memarg.add_argument("--history_length", type=int, default=4, help="Number of most recent frames experiences by the agent.")

dqnarg = parser.add_argument_group('DQN')
dqnarg.add_argument("--lr", type=float, default=0.00025, help="Learning rate.")
dqnarg.add_argument("--lr_anneal", type=float, default=20000, help="Step size of learning rate annealing.")
dqnarg.add_argument("--discount", type=float, default=0.99, help="Discount rate.")
dqnarg.add_argument("--batch_size", type=int, default=32, help="Batch size.")
dqnarg.add_argument("--accumulator", type=str, default='mean', help="Batch accumulator.")
dqnarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp.")
dqnarg.add_argument("--min_decay_rate", type=float, default=0.01, help="Min decay rate for RMSProp.")
dqnarg.add_argument("--init_eps", type=float, default=1.0, help="Initial value of e in e-greedy exploration.")
dqnarg.add_argument("--final_eps", type=float, default=0.1, help="Final value of e in e-greedy exploration.")
dqnarg.add_argument("--final_eps_frame", type=float, default=1000000, help="The number of frames over which the initial value of e is linearly annealed to its final.")
dqnarg.add_argument("--clip_delta", type=float, default=1, help="Clip error term in update between this number and its negative.")
dqnarg.add_argument("--steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
dqnarg.add_argument("--train_steps", type=int, default=500000, help="Number of training steps.")
dqnarg.add_argument("--update_freq", type=int, default=4, help="The number of actions selected between successive SGD updates.")
dqnarg.add_argument("--replay_start_size", type=int, default=50000, help="A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory.")
dqnarg.add_argument("--save_weights", type=int, default=10000, help="Save the mondel after this many steps.")

testarg = parser.add_argument_group('Test')
testarg.add_argument("--display", dest="display", help="Display screen during testing.")
testarg.set_defaults(display=False)
testarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of no-op actions to be performed by the agent at the start of an episode.")
testarg.add_argument("--ckpt_dir", type=str, default='model', help="Tensorflow checkpoint directory.")
testarg.add_argument("--out", help="Output directory for gym.")
testarg.add_argument("--episodes", type=int, default=100, help="Number of episodes.")
testarg.add_argument("--seed", type=int, help="Random seed.")
args = parser.parse_args()
if args.seed:
    rand.seed(args.seed)
if not os.path.exists(args.ckpt_dir):
	os.makedirs(args.ckpt_dir)

#Checking for/Creating gym output directory
if args.out:
	if not os.path.exists(args.out):
		os.makedirs(args.out)
else:
	if not os.path.exists('gym-out/' + args.game):
		os.makedirs('gym-out/' + args.game)
	args.out = 'gym-out/' + args.game

##here we go...

# initialize gym environment and dqn
env = Environment(args)
agent = DQN(env, args)

# train agent
Trainer(agent).run()

# play the game
env.gym.monitor.start(args.out, force=True)
agent.play()
env.gym.monitor.close()
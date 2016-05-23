Overview
====
A Tensorflow DQN implementation based on [DeepMind's DQN](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) for playing Atari games.This the code for 'Build a game AI' on [Youtube](https://youtu.be/HBAUeJkFMH0)


## Dependencies
- [gym](https://gym.openai.com)
- [Tensorflow](https://www.tensorflow.org)
- [virtualenv] (https://virtualenv.pypa.io/en/latest/installation.html)

## Basic Usage
To run, type the following into terminal

`python atari.py --game <env_name>`

It will run `SpaceInvaders-v0` by default but you can use other game names as well. Add `--display true` to the above command line argument if you'd like to see the game while it trains

## Credits
Credit for the vast majority of code here goes to [Kee Hyun Won](https://github.com/kihyunwon). I've merely created a wrapper to get people started.

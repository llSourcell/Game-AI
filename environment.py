import gym
import cv2

class Environment:

    def __init__(self, params):
        self.gym = gym.make(params.game)
        self.observation = None
        self.display = params.display
        self.terminal = False
        self.dims = (params.height, params.width)

    def actions(self):
        return self.gym.action_space.n

    def restart(self):
        self.observation = self.gym.reset()
        self.terminal = False

    def act(self, action):
        if self.display:
            self.gym.render()
        self.observation, reward, self.terminal, info = self.gym.step(action)
        if self.terminal:
            #if self.display:
            #    print "No more lives, restarting"
            self.gym.reset()
        return reward

    def getScreen(self):
        return cv2.resize(cv2.cvtColor(self.observation, cv2.COLOR_RGB2GRAY), self.dims)

    def isTerminal(self):
        return self.terminal


from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import random
from collections import deque
from agent import DDPGAgent

class Trainer(object):
    def __init__(self, environment, agent_factory, tracker_factory, seed=1):
        self.env = environment
        self.seed = seed
        self.tracker_factory = tracker_factory
        self.agent_factory = agent_factory

        # Environment information:
        self.brain_name = environment.brain_names[0]
        self.brain = environment.brains[self.brain_name]

        test_observation = environment.reset(train_mode=True)[self.brain_name]
        initial_state = test_observation.vector_observations
        self.state_size = initial_state.shape[1]

        self.num_agents = len(test_observation.agents)
        self.action_size = self.brain.vector_action_space_size

        self.sample_state = initial_state[0]

    def describe_environment(self):
        # print (str(env))
        dir(self.env)
        print("Seed: ", self.seed)
        print("Brain names: ", self.env.brain_names)
        print (self.env.brains)
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('Type of action space:', self.brain.vector_action_space_type)
        print('There are {} agents. Each observes a state with length: {} and takes an action of size {}'.format(self.num_agents, self.state_size, self.action_size))
        print('Type of state space:', self.brain.vector_observation_space_type)
        print('The state for the first agent looks like:', self.sample_state[0])

    def train_single(self, n_episodes=1000, plot_every=10):
        tracker = self.tracker_factory.createTracker(n_episodes)
        agent = self.agent_factory.createAgent(state_size=self.state_size, action_size=self.action_size, random_seed = self.seed)
        env, observation, brain_name = self.env, self.env.reset(train_mode=True)[self.brain_name], self.brain_name
        state = observation.vector_observations[0]
        agent.load()
        for i_episode in range(1, n_episodes+1):
            agent.reset_episode()
            while True:
                action = agent.act(state)
                observation = env.step([action])[brain_name]
                next_state, reward, done = observation.vector_observations[0], observation.rewards[0], observation.local_done[0]
                agent.step(state, action, reward, next_state, done)
                tracker.step(i_episode, reward, done)
                state = next_state
                if done:
                    break
            tracker.print_episode_summary()
            if i_episode % plot_every == 0:
                tracker.plot_performance()

        agent.save()

    """
        Player for multi-agent game
    """
    def play_multiple(self):
        env, num_agents, action_size, brain_name = self.env, self.num_agents, self.action_size, self.brain_name
        observation = env.reset(train_mode=False)[brain_name]      # reset the environment    
        states = observation.vector_observations                   # get the current state (for each agent)
        scores = np.zeros(num_agents)                              # initialize the score (for each agent)
        while True:
            actions = agent.act(states)                            # select an action (for each agent)
            observation = env.step(actions)[brain_name]               # send all actions to tne environment
            next_states, rewards, dones = observation.vector_observations, observation.rewards, observation.local_done
            scores += rewards                                      # update the score (for each agent)
            states = next_states                                   # roll over states to next time step
            if np.any(dones):                                      # exit loop if episode finished
                break
        print('Total score (averaged over {} agents) this episode: {}'.format(num_agents, np.mean(scores)))

    """
        Player for multi-agent game
    """
    def play_single(self):
        env, action_size, brain_name = self.env, self.action_size, self.brain_name
        observation = env.reset(train_mode=False)[brain_name]       # reset the environment    
        state = observation.vector_observations[0]                  # get the current state (for each agent)
        score = 0                                                   # initialize the score (for each agent)
        while True:
            action = agent.act(state)                               # select an action (for each agent)
            observation = env.step(action)[brain_name]              # send all actions to tne environment
            next_state, reward, done = observation.vector_observations[0], observation.rewards[0], observation.local_done[0]
            score += reward                                         # update the score (for each agent)
            state = next_state                                      # roll over states to next time step
            if done:                                                # exit loop if episode finished
                break
        print('Total score for this episode: {}'.format(score))

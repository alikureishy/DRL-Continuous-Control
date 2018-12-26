import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

AVERAGING_WINDOW = 100

class TrackerFactory(object):
    def __init__(self):
        pass
    
    def createTracker(self, num_episodes):
        return PerformanceTracker(num_episodes)

class PerformanceTracker(object):
    
    def __init__(self, num_episodes):
        """Constructor
        
        Params
        ======
            num_episodes (int): the number of episodes that are to be tracked
        """
        self.scores = np.zeros(num_episodes, dtype=np.float32)
        self.positives = np.zeros(num_episodes, dtype=np.float32)
        self.positive_counts = np.zeros(num_episodes, dtype=np.float)
        self.negatives = np.zeros(num_episodes, dtype=np.float32)
        self.negative_counts = np.zeros(num_episodes, dtype=np.float32)
        self.neutral_counts = np.zeros(num_episodes, dtype=np.float32)
        self.step_counts = np.zeros(num_episodes, dtype=np.float32)
        self.accuracies = np.zeros(num_episodes, dtype=np.float32)
        self.recalls = np.zeros(num_episodes, dtype=np.float32)
        self.centennial_averages = np.zeros(num_episodes, dtype=np.float32)
        self.episode_durations = np.zeros(num_episodes, dtype=np.int32)
        self.num_episodes = num_episodes
        self.train_start_time = None
        self.train_duration = None
        self.episode_start_time = None
        self.current_episode = 0
        
    def started_training(self):
        self.train_start_time = time.time()

    def started_episode(self, episode):
        self.episode_start_time = time.time()

    def step(self, episode, reward, done):
        """Save reward of a step taken for a given episode 
        
        Params
        ======
            episode (int): the episode number
            rewards numpy(int): rewards obtained on the last step
            dones numpy(bool): what the status is of each of the agents
        """
        assert episode >= 0 and episode < self.num_episodes, "Invalid episode number: {}".format(episode)

        self.current_episode = episode
        self.step_counts[episode] += 1
        if reward < 0.0:
            self.negatives[episode] += 1.0 * reward
            self.negative_counts[episode] += 1
            # print ("<({:.3f})".format(reward), end="")
        elif reward == 0.0:
            self.neutral_counts[episode] += 1
            # print (".", end="")
        elif reward > 0.0:
            self.positives[episode] += 1.0 * reward
            self.positive_counts[episode] += 1
            # print (">({:.3f})".format(reward), end="")

        if done:
            self.scores[episode] = self.positives[episode] + self.negatives[episode]
            self.accuracies[episode] = (1.0 * self.positive_counts[episode]) / self.step_counts[episode]

            if episode >= AVERAGING_WINDOW:
                self.centennial_averages[episode] = np.mean(self.scores[episode-AVERAGING_WINDOW:episode])
            else:
                self.centennial_averages[episode] = np.mean(self.scores[0:episode])
            # print("|\n")
        # else:
            # if (self.step_counts[episode] % 100 == 0):
            #     print("\n")

    def ended_episode(self, episode, print_episode_summary=False):
        duration = time.time() - self.episode_start_time
        self.episode_durations[episode] = duration
        if print_episode_summary:
            self.print_episode_summary(episode)
    
    def ended_training(self):
        self.train_duration = self.train_start_time - time.time()

    def get_mean_centennial_score(self, episode=None):
        if episode == None:
            episode = self.current_episode
        return self.centennial_averages[episode]

    # TODO:
    def get_agent_mean_score(self, episode):
        return self.scores[episode]

    # TODO:
    def get_agent_max_score(self, episode):
        return self.scores[episode]

    # TODO:
    def get_agent_min_score(self, episode):
        return self.scores[episode]

    # TODO:
    def get_episode_duration(self, episode):
        return self.episode_durations[episode]

    def get_training_duration(self):
        return self.train_duration

    def plot_performance(self):
        """Plot various statistics captured by the tracker
        """
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Accuracy", episodes, "Episode", self.accuracies, "Accuracy", id=311)
        self.__plot__("Averaged Episode Scores", episodes, "Episode", self.scores, "Score", id=312) 
        self.__plot__("Centennial Averages", episodes, "Episode", self.centennial_averages, "{}-Episode Average Score".format(AVERAGING_WINDOW), id=321)
        self.__plot__("Episode Step Counts", episodes, "Episode", self.step_counts, "# steps", id=322)
        self.__plot__("Episode Durations", episodes, "Episode", self.episode_durations, "Duration (secs)", id=331)
        plt.show()
            
    def print_episode_summary(self, episode):
        i = episode
        print('\rEpisode :: {}\tScores:\tCentennial: {:.3f}\tMean: {:.3f}\tMin: {:.3f}\tMax:{:.3f}\tDuration: {:.2f}s'
                .format(i, self.get_mean_centennial_score(i), self.get_agent_mean_score(i), 
                            self.get_agent_min_score(i), self.get_agent_max_score(i), self.get_episode_duration(i)))
        # print('\tCompleted in {} steps with {} +ve, {} -ve and {} neutral, rewards. Episode score: {:.3f}. Accuracy: {:.3f}'
        #     .format(self.step_counts[i], self.positive_counts[i], self.negative_counts[i], 
        #         self.neutral_counts[i], self.scores[i], self.accuracies[i]))

    def __plot__(self, title, xvalues, xlabel, yvalues, ylabel, id=111):
        """Generic plot utility to plot the given values and labels
        
        Params
        ======
            title (string): graph title
            xvalues (list): list of values for the x-axis
            xlabel (string): label of the x-axis
            yvalues (list): list of values for the y-axis
            ylabel (string): label of the y-axis
        """
        fig = plt.figure()
        ax = fig.add_subplot(id)
        ax.plot(xvalues, yvalues)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
#         ax.grid()
#         fig.savefig("test.png")


        # last_100_scores = deque(maxlen=100)
        # training_score_history = []                          # initialize the score (for each agent)
        #     episode_score = 0
        #     print("\t")

        #     last_100_scores.append(episode_score)
        #     training_score_history.append(episode_score)
        #     last_100_scores_avg = np.mean(last_100_scores)

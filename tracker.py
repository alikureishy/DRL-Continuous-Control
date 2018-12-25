import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
        self.scores = np.zeros(num_episodes, dtype=np.int32)
        self.positives = np.zeros(num_episodes, dtype=np.int32)
        self.positive_counts = np.zeros(num_episodes, dtype=np.int32)
        self.negatives = np.zeros(num_episodes, dtype=np.int32)
        self.negative_counts = np.zeros(num_episodes, dtype=np.int32)
        self.neutral_counts = np.zeros(num_episodes, dtype=np.float32)
        self.step_counts = np.zeros(num_episodes, dtype=np.int32)
        self.accuracies = np.zeros(num_episodes, dtype=np.float32)
        self.recalls = np.zeros(num_episodes, dtype=np.float32)
        self.running_averages = np.zeros(num_episodes, dtype=np.float32)
        self.num_episodes = num_episodes
        
    def step(self, episode, reward, done):
        """Save reward of a step taken for a given episode 
        
        Params
        ======
            episode (int): the episode number
            reward (int): reward obtained on the last step
            done (bool): whether this was the last step of the episode
        """
        assert episode >= 0 and episode < self.num_episodes, "Invalid episode number: {}".format(episode)

        self.step_counts[episode] += 1
        if reward < 0:
            self.negatives[episode] += reward
            self.negative_counts[episode] += 1
        elif reward == 0:
            self.neutral_counts[episode] += 1
        elif reward > 0:
            self.positives[episode] += reward
            self.positive_counts[episode] += 1

        if done:
            self.scores[episode] = self.positives[episode] + self.negatives[episode]
            self.accuracies[episode] = (1.0 * self.positive_counts[episode]) / (self.positive_counts[episode] + 
                                                                                self.negative_counts[episode])

            if episode >= AVERAGING_WINDOW:
                self.running_averages[episode] = np.mean(self.scores[episode-AVERAGING_WINDOW:episode])
            else:
                self.running_averages[episode] = np.mean(self.scores[0:episode])

    def plot_performance(self):
        """Plot various statistics captured by the tracker
        """
        episodes = np.arange(1, self.num_episodes+1)
        self.__plot__("Accuracy", episodes, "Episode", self.accuracies, "Accuracy", id=411)
        self.__plot__("Scores", episodes, "Episode", self.scores, "Score", id=412) 
        self.__plot__("{}-Episode Running Averages".format(AVERAGING_WINDOW), episodes, "Episode", self.running_averages, "{}-Episode Average Score".format(AVERAGING_WINDOW), id=413)
        self.__plot__("Durations", episodes, "Episode", self.step_counts, "Duration (# steps)", id=314)
        plt.show()
            
    def print_episode_summary(self, episode):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, self.running_averages[episode]), end="")

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

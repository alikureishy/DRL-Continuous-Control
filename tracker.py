from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

AVERAGING_WINDOW = 100

class TrackerFactory(object):
    def __init__(self):
        pass
    
    def createTracker(self, num_episodes, num_agents):
        return PerformanceTracker(num_episodes, num_agents)

class PerformanceTracker(object):
    
    def __init__(self, num_episodes, num_agents):
        """Constructor
        
        Params
        ======
            num_episodes (int) : the number of episodes that are to be tracked
            num_agents (int)   : the number of agent steps being calculated in parallel
        """

        # Scalars:
        self.num_episodes = num_episodes
        self.num_agents = num_agents
        self.training_start_time = None
        self.training_duration = None
        self.last_episode = 0
        self.last_episode_starttime = None

        # 1-D Vectors: [Num_Episodes]
        self.temporal_mean_scores   = np.zeros(num_episodes, dtype=np.float32)
        self.temporal_min_scores    = np.zeros(num_episodes, dtype=np.float32)
        self.temporal_max_scores    = np.zeros(num_episodes, dtype=np.float32)
        self.temporal_durations     = np.zeros(num_episodes, dtype=np.float32)
        self.temporal_step_counts   = np.zeros(num_episodes, dtype=np.float32)
        self.centennial_scores      = np.zeros(num_episodes, dtype=np.float32)
        self.temporal_score_window   = deque(maxlen=100)

        # 2-D Matrices: Num_Episodes X Num_Agents
        self.score_table            = np.zeros((num_episodes, num_agents), dtype=np.float32)
        
    def started_training(self):
        self.training_start_time = time.time()

    def started_episode(self, episode):
        self.last_episode = episode
        self.last_episode_starttime = time.time()

    def step(self, episode, rewards, dones):
        """Save reward of a step taken for a given episode 
        
        Params
        ======
            episode (int): the episode number
            rewards numpy(int): rewards obtained on the last step
            dones numpy(bool): what the status is of each of the agents
        """
        assert episode >= 0 and episode < self.num_episodes, "Invalid episode number: {}".format(episode)
        assert episode == self.last_episode, "Did you forget to call started_episode() or ended_episode()?"

        # Register steps & rewards:
        self.temporal_step_counts[episode] += 1
        self.score_table[episode] += rewards

        # End-of-episode book-keeping:
        if np.any(dones):
            self.temporal_mean_scores[episode] = np.mean(self.score_table[episode])
            self.temporal_min_scores[episode] = np.min(self.score_table[episode])
            self.temporal_max_scores[episode] = np.max(self.score_table[episode])
            self.temporal_score_window.append(self.temporal_mean_scores[episode])
            self.centennial_scores[episode] = np.mean(self.temporal_score_window)
            # print("|\n")
        # else:
            # if (self.step_counts[episode] % 100 == 0):
            #     print("\n")

    def print_episode_summary(self, episode):
        i = episode
        # print (i, self.last_episode)
        print('\rEpisode :: {}\tScores:\tCentennial: {:.3f}\tMean: {:.3f}\tMin: {:.3f}\tMax:{:.3f}\tDuration: {:.2f}s'
                .format(i, self.get_centennial_score(i), self.get_temporal_mean_score(i), 
                            self.get_temporal_min_score(i), self.get_temporal_max_score(i), self.get_temporal_duration(i)))

    def ended_episode(self, episode, print_episode_summary=False):
        assert episode == self.last_episode, "Did you forget to call started_episode() or ended_episode()?"
        self.temporal_durations[episode] = time.time() - self.last_episode_starttime
        if print_episode_summary:
            self.print_episode_summary(episode)
    
    def ended_training(self):
        self.training_duration = self.training_start_time - time.time()

    def get_centennial_score(self, episode=None):
        if episode == None:
            episode = self.last_episode
        return self.centennial_scores[episode]

    def get_temporal_mean_score(self, episode):
        return self.temporal_mean_scores[episode]

    def get_temporal_max_score(self, episode):
        return self.temporal_max_scores[episode]

    def get_temporal_min_score(self, episode):
        return self.temporal_min_scores[episode]

    def get_temporal_duration(self, episode):
        return self.temporal_durations[episode]

    def get_training_duration(self):
        return self.training_duration

    def plot_performance(self):
        """Plot various statistics captured by the tracker
        """
        i = self.last_episode
        if (i >= 1):
            episodes = np.arange(0, i+1) #.reshape((1, i))
            self.__plot__("Averaged Episode Scores", episodes, "Episode", self.temporal_mean_scores[:i+1], "Score", id=311) 
            self.__plot__("Episode Step Counts", episodes, "Episode", self.temporal_step_counts[:i+1], "# steps", id=312)
            self.__plot__("Episode Durations", episodes, "Episode", self.temporal_durations[:i+1], "Duration (secs)", id=321)
            self.__plot__("Centennial Averages", episodes, "Episode", self.centennial_scores[:i+1], "Score", id=322)
            plt.show()
            
    def __plot__(self, title, xs, xlabel, ys, ylabel, id=111):
        """Generic plot utility to plot the given values and labels
        
        Params
        ======
            title (string): graph title
            xs (list): list of values for the x-axis
            xlabel (string): label of the x-axis
            ys (list): list of values for the y-axis
            ylabel (string): label of the y-axis
        """
        # if (len(xs.shape) == 2 and len(ys.shape) == 2):
        fig = plt.figure()
        ax = fig.add_subplot(id)
        # print("Xs: ", xs.shape, "\tYs: ", ys.shape)
        ax.plot(xs, ys)
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


from unityagents import UnityEnvironment
from trainer import Trainer
from tracker import PerformanceTracker
from agent import AgentFactory
from tracker import TrackerFactory
import time


def main():
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

    agent_factory = AgentFactory()
    tracker_factory = TrackerFactory()

    trainer = Trainer(env, agent_factory, tracker_factory)
    trainer.describe_environment()

    tracker = trainer.train_single(n_episodes=1000, plot_every=10)
    tracker.plot_performance()

    time.sleep(10000)


if __name__ == '__main__':
    main()
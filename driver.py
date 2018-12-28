
from unityagents import UnityEnvironment
from trainer import Trainer
from tracker import PerformanceTracker
from agent import AgentFactory
from tracker import TrackerFactory
import time

def main():
    print("Training the agent ...")
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    agent_factory = AgentFactory()
    tracker_factory = TrackerFactory()
    trainer = Trainer(env, agent_factory, tracker_factory)
    trainer.describe_environment()
    agent, tracker = trainer.train(n_episodes=200, plot_every=1000, learn_every=20, iterations_per_learn=10, goal_score=30.0)
    # tracker.plot_performance()
    # env.close()
    print("Training complete!")

    print("Running the trained agent ...")
    # env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    trainer.play()
    time.sleep(10)
    env.close()
    print("Game finished!")

    time.sleep(5)



if __name__ == '__main__':
    # import cProfile
    # cp = cProfile.Profile()
    # cp.enable()

    main()

    # cp.disable()
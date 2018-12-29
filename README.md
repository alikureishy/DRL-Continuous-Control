

## Overview


## Environment

As mentioned in SETUP.md, the environment being solved is that of ______.

The environment has 20 agents, each contributing an independent observation to the training process. The reason for having 20 agents was to gather more diverse experiences tuples, so that the agents can learn from each other's experiences. As you will see in the training section, the results were quite promising. The same algorithm would take far longer to train using a single agent, primarily because of the lack of exploratory experiences.

I used a GPU (NVidia GeForce Titan Xp) for running most of the tensor-based operations of PyTorch, which greatly sped up the learning process.

A requirements.txt file has been included for reproducing the appropriate conda environment to run this system.


## Design

I have structured the solution using the below components (classes).

### class Driver
This is the command-line entry-point to launch the trainign system. The goal was to mimic the IPynb Notebook environment without having to use the notebook, which greatly facilitated development and testing. 

### class Trainer
Encapsulates the train() and play() routines.

### class Tracker
Encapsulates, collects and tracks various metrics through the training process. 

### class Agent
Encapsulates the processing of observation and the output of associated actions -- i.e, the 'agent'.

#### class Actor (DNN)
Encapsulates the policy-based learning (high variance, low bias) component of the DDPG algorithm for DRL.

#### class Critic (DNN)
Encapsulates value-based supervision (low variance, low bias) of the policy-based learner (Actor).

### Hyperparameters

I found that for the soft-update, setting the value of TAU to be 1e-2 trained pretty well.

#### GAMMA

I set GAMMA to 0 because increasing this value would produce homogeneity in the learning process across the agents, which defeated the purpose of having multiple "agents" in the environment.


## Training & Results

As the graphs below illustrate, the agent learned the problem space pretty well, achieving the goal score in a little over 100 episodes.




## Future Improvements

### Actions as Gaussian Distributions

### Prioritized Experience Replay


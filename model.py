import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Layer 1
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Layer 2
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Layer 3       
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(self.bn1(state))
        x = F.relu(self.fc2(self.bn2(x))

        # tanh produces a real-valued output R[-1, +1] that allows 
        # continuous action spaces
        out = F.tanh(self.fc3(self.bn3(x)))
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        # Since for Q-values, we have to utilize the selected action too
        # That (state-action) pair is obtained after the state is put through
        # one hidden layer and then concatenated with the selected action
        # during evaluation
        fcs1_s_a_units = fcs1_units + action_size;

        self.seed = torch.manual_seed(seed)

        # Action-normalizer:
        #   This may not be necessary, but I'm adding it here to support a
        #   model-free learner, just in case the action tensor has values 
        #   that are outside the [-1,+1] case. This then makes it appropriate
        #   to be concatenated with the batch-normed state-input (at the 
        #   2nd layer = fcs2)
        self.an1 = nn.BatchNorm1d(action_size)

        # Layer 1
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)

        # Layer 2
        self.bn2 = nn.BatchNorm1d(state_size)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)

        # Layer 3 - a reward is a single-dimensional real-valued number
        #   Expressing the reward as a multi-dimension tensor would
        #   perhaps allow the network to learn a more nuanced reward mechanism
        self.bn3 = nn.BatchNorm1d(state_size)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
            Note that the 'action' parameter here is a continuous-valued action space
            tensor, which implies that it has a value between [-1, +1].
            It needs to be batch-normed as well.
        """
        state = 
            F.relu(self.fc1(self.bn1(state))
        action =
            
        x = torch.cat((inputs, action), dim=1)
        x = F.relu(self.fc2(self.bn2(x))

        # The value-function is a real-valued number that is no constrained
        # any range, as is the case for continuous action selection.
        # This is because here we are not selecting an 'action', but rather, 
        # we're learning to approximate the reward function (i.e, to predict the 
        # expected reward, which is just an unconstrained real valued number R[-inf, +inf]
        out = self.fc3(self.bn3(x))

        # I we were to want to clip rewards, this may be the place to clip them.
        # out = clip(out)

        return out

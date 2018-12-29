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
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        # Layer 2
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Layer 3       
        self.fc3 = nn.Linear(fc2_units, action_size) # Policy output expressed as action vector

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state

        # x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # tanh produces a real-valued output R[-1, +1] that allows 
        # continuous action spaces
        # out = self.bn3(x)
        out = x
        out = self.fc3(out)
        out = F.tanh(out)
        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, reward_size=1):
        """Initialize parameters and build model.
            For Q-values, we need both the state and the action.
            That (state-action) pair is obtained after the state is put through
            one hidden layer and then concatenated with the action selected by
            the Actor during evaluation.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Layer 1
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        # Layer 2 (merge with action inputs)
        merged_size = fc1_units+action_size;
        self.fc2 = nn.Linear(merged_size, fc2_units)

        # Layer 3 - a reward is a single-dimensional real-valued number
        #   Expressing the reward as a multi-dimension tensor would
        #   perhaps allow the network to learn a more nuanced reward mechanism
        self.fc3 = nn.Linear(fc2_units, reward_size)  # The reward vector of REAL(-inf, +inf) numbers. Default is just size 1.

        self.reset_parameters()

    def reset_parameters(self):
        # self.action_fc1.weight.data.uniform_(*hidden_init(self.action_fc1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
            Note that the 'action' parameter here is a continuous-valued action space
            tensor, which implies that it has a value between [-1, +1].
            It needs to be batch-normed as well.
        """
        state  = self.fc1(state)
        state  = self.bn1(state)
        state  = F.relu(state)

        # Merge action_input and state_input  
        x = torch.cat((state, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)

        # The value-function is a real-valued number that is no constrained
        # any range, as is the case for continuous action selection.
        # This is because here we are not selecting an 'action', but rather, 
        # we're learning to approximate the reward function (i.e, to predict the 
        # expected reward, which is just an unconstrained real valued number R[-inf, +inf]
        out = x
        out = self.fc3(out)

        # I we were to want to clip rewards, this may be the place to clip them.
        # out = clip(out)

        return out

# For the Critic initializer:
# Action-normalizer:
#   This may not be necessary, but I'm adding it here to support a
#   model-free learner, just in case the action tensor has values 
#   that are outside the [-1,+1] case. This then allows it
#   to be concatenated with the batch-normed state-input (at the 
#   2nd layer = fcs2)
# self.action_bn1 = nn.BatchNorm1d(action_size)
# self.action_fc1 = nn.Linear(action_size, action_fc1_units) # Right now this just gets another vector of action_size


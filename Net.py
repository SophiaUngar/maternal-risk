# Sophia Ungar

import torch
from coral_pytorch.layers import CoralLayer

class NeuralNet(torch.nn.Module):

    # constructs the net
    def __init__(self, input_layer_size, hidden_layer_size, num_classes):  # , *args, **kwargs
        super(NeuralNet, self).__init__()

        self.num_classes = num_classes

        # 2 layers with 6 nodes each at the moment
        self.fc1 = torch.nn.Linear(input_layer_size, hidden_layer_size)  # fc means fully connected
        self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)  # out first layer has to match in second

        # coral layer
        self.fc3 = CoralLayer(size_in=hidden_layer_size, num_classes=num_classes - 1)

        # TODO: what numbers do we want in the middle here?
        # TODO: do we want more layers?

    # forward propagation
    def forward(self, x):
        # x = self.model(x)
        # return x
        x = torch.nn.functional.relu(self.fc1(x))  # pass through layer 1
        x = torch.nn.functional.relu(self.fc2(x))  # pass through layer 2
        x = self.fc3(x)
        # do I need F.relu around the first function here? something else?
        # yes, you should
        # it is an activation function -
        # that helps learn patterns without too much complication
        # https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
        return x


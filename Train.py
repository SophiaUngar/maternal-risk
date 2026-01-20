# Sophia Ungar

import Net

import torch
from coral_pytorch.losses import corn_loss


def train(x,y,epochs,hidden_layer_size, num_classes):
    # TODO: is this an appropriate function?

    model = Net.NeuralNet(x.size(1),hidden_layer_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for rounds in range(epochs):

        # forward propagation
        output = model(x)

        loss = corn_loss(x, y, model.num_classes)

        # backwards propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    return model
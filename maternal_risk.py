# Sophia Ungar

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# I found the following helpful:
# Torch documentation + tutorial
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
# https://www.geeksforgeeks.org/getting-started-with-pytorch/
# https://www.kdnuggets.com/a-beginners-guide-to-pytorch 


class NeuralNet(nn.Module):
    # constructs the net
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc = nn.Linear(10,5) # fully connected
    
    def forward(self,x):
        return self.fc(x)
        


# returns ordinal data 
def to_ordinal(phrase):
    num = 2

    if phrase=='high risk':
        num = 1
    elif phrase=='mid risk':
        num = 0
    elif phrase=='low risk':
        num = -1
    else:
        num = 2
        # TODO: throw error
    return num


def process(filename, y_name):
    # read in csv and turn it into a numpy array
    data = pd.read_csv(filename)
    indices = list(data.columns)
    data = data.to_numpy()
    y_index = list(indices).index(y_name)

    # shuffles the points
    np.random.shuffle(data)

    # split into x and y
    x = np.delete(data, y_index, 1).astype(np.float64) # everything except y
    y = data[:,y_index] # grabs the y_index column

    # code string data into numeric
    y = np.vectorize(to_ordinal)(y).astype(np.float64)#.to_frame() 
    
    # turn into tensors and return
    return torch.as_tensor(x), torch.as_tensor(y)
# things I learned:
# np.delete(frame, index, axis)
# astype because it could not infer what the type of the np was


def train(x,y):
    pass

def test(x,y):
    pass


def main():
    filename = "Data\Maternal Health Risk Data Set.csv"
    y_name = 'RiskLevel'

    x,y = process(filename, y_name)



    train(x,y)



    print("Done!")


main()


# improvements
# - send data to devices with cuda:0
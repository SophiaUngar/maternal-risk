# Sophia Ungar

from Load_Data import process
from Train import train
from Test import test

import torch


def main():
    filename = "Data/Maternal Health Risk Data Set.csv" # data filename
    y_name = 'RiskLevel' # name of category we are trying to predict
    torch.manual_seed(21)

    # load data and prepare it for training
    x,y = process(filename, y_name)
    num_classes = torch.unique(y, dim = 0).size(dim=0)

    training_size = 1000 # portion of the data used for training versus testing
    data_train_x, data_train_y = x[:training_size],y[:training_size]
    data_test_x, data_test_y = x[training_size:],y[training_size:]

    # training
    epochs = 100
    hidden_layer_size = 10
    model = train(data_train_x,data_train_y, epochs, hidden_layer_size, num_classes)

    # testing
    accuracy = test(data_test_x, data_test_y, model)
    print("Accuracy: ", accuracy)

    print("Done!")


main()


# TODO improvements
# - send data to devices with cuda:0?
# - do I need more data transformations?
# - what loss function do I want, and is the computer using it appropriately?
# - make sure that it can work with other data without too much fuss

# what I've learned
# - fashion-mnist is a data set, not a data structure

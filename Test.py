# Sophia Ungar

from coral_pytorch.dataset import corn_label_from_logits


# returns how often the model is right
def test(X,y, model):
    estimated_y = model(X)
    predicted_labels = corn_label_from_logits(X)

    accurate_values = 0

    for y_val in range(y.shape[0]):
        if (y[y_val] == predicted_labels[y_val]):
            accurate_values += 1

    accuracy = accurate_values/y.shape[0]

    return accuracy

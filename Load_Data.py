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
    x = np.delete(data, y_index, 1)#.astype(np.float64)  # everything except y
    y = data[:, y_index]  # grabs the y_index column

    # normalize
    # x = torch.nn.functional.normalize(torch.FloatTensor(x))

    # code string data into numeric
    y = np.vectorize(to_ordinal)(y) #.astype(np.float64)  # .to_frame()
    for val in range(y.size):
        y[val] += 1

    # turn into tensors and return
    return torch.Tensor(x.astype(np.float64)), torch.LongTensor(y)
# things I learned:
# np.delete(frame, index, axis)
# astype because it could not infer what the type of the np was
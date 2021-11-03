import numpy as np

def sigmoid(drive):                                                 # sigmoid activation func
    return 1 / (1 + np.exp(-drive))

def sigmoidprime(drive):                                            # derivative of the sigmoid activation func
    exp = np.exp(drive)
    return exp / (exp + 1)**2

def squared_error(t, y):                                            # loss func (qualitative performance measure)
    return (t - y)**2

def accuracy(t, y, threshold = 0.5):                                 # quantitative performance measure
    return np.mean(np.abs(t - y) <= threshold)


if __name__ == "__main__":
    print(sigmoid(7))
    print(sigmoidprime(7))
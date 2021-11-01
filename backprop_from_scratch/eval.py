import numpy as np

def sigmoid(drive):                                                 # sigmoid activation func
    return 1 / (1 + np.exp(-drive))

def sigmoidprime(drive):                                            # derivative of the sigmoid activation func
    sig = sigmoid(drive)
    return sig * (1 - sig)

def squared_error(t, y):                                            # loss func (qualitative performance measure)
    return (t-y)**2

def accuracy(t, y, theshold = 0.5):                                 # quantitative performance measure
    labels = np.where(t == True, y - theshold, y + theshold)
    true_labels = labels[0 < labels < 1]
    return true_labels/labels



if __name__ == "__main__":
    print(sigmoid(7))
    print(sigmoidprime(7))
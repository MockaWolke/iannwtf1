import numpy as np

def sigmoid(drive):                            # sigmoid activation func
    return 1 / (1 + np.exp(-drive))

def sigmoidprime(drive):                       # derivative of the sigmoid activation func
    sig = sigmoid(drive)
    return sig * (1 - sig)

if __name__ == "__main__":
    print(sigmoid(7))
    print(sigmoidprime(7))

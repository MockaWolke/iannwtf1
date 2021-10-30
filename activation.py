import numpy as np

def sigmoid(d):                            # sigmoid activation func
    return 1 / (1 + np.exp(-d))

def sigmoidprime(d):                       # derivative of the sigmoid activation func
    sig = sigmoid(d)
    return sig * (1 - sig)

if __name__ == "__main__":
    print(sigmoid(7))
    print(sigmoidprime(7))

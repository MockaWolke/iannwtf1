from mlp import MLP
from dataset import INPUTS

ann = MLP(depth = 5)

prediction = ann.forward_step(INPUTS[0])
print(ann.activations)

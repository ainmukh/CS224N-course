from utils.utils import softmax
import numpy as np

a = np.array([1, 2, 3])
print(np.exp(a) / (1 + np.exp(a)))

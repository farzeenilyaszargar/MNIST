import numpy as np


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]);
weights = np.array([[0.5], [0.5]]);
bias = 0.5;

z = np.dot(inputs, weights) + bias;

output = np.maximum(z, 0); #relu

print("Output: ", output);
print("Z: ", z);



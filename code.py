import numpy as np
import matplotlib.pyplot as plt
from openai import images

def softmax(x):
    expValues = np.exp(x)
    return expValues / np.sum(expValues) 


# process data (read image from 28x28 array to 784 pixel values as a big array)

# get images loaded here as an array of 784 values
# images = list of images (each image is a 784 pixel array)



# layer 1: 784 -> 128

weights1 = np.random.rand(784, 128)
biases1 = np.random.rand(128)

# layer 2: 128 -> 10

weights2 = np.random.rand(128, 10)
biases2 = np.random.rand(10)




# run loop to train the model i.e get the weights and biases for the model

for epoch in range(10):  # number of epochs
    totalLoss = 0
    for i in range(len(images)):
        # forward pass
        image = images[i]
        label = labels[i]  

        z1 = np.dot(images, weights1) + biases1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, weights2) + biases2


        prob = softmax(z2); 


        loss = -np.log(prob[label]) 
        totalLoss += loss
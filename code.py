import numpy as np
from torchvision.datasets import MNIST



def softmax(x):
    expValues = np.exp(x - np.max(x))
    return expValues / np.sum(expValues) 


# process data (read image from 28x28 array to 784 pixel values as a big array)

# get images loaded here as an array of 784 values
# images = list of images (each image is a 784 pixel array)

train_data = MNIST(
    root="./data",
    train=True,
    download=False
)

test_data = MNIST(
    root="./data",
    train=False,
    download=False
)

images = train_data.data.numpy().reshape(-1, 28 * 28)   # flatten the images to 784 pixel values
labels = train_data.targets.numpy()

test_images = test_data.data.numpy().reshape(-1, 28 * 28)  # flatten the images to 784 pixel values
test_labels = test_data.targets.numpy()


# flatten the images to 784 pixel values in terms of brightness values between 0 and 1 (normalize the pixel values)

images = images.astype(np.float32) / 255.0 
test_images = test_images.astype(np.float32) / 255.0

print(f"Number of training images: {len(images)}")
print(f"Number of test images: {len(test_images)}")
print("Sample image shape:", images[0].shape)

# layer 1: 784 -> 128

weights1 = np.random.rand(784, 128)
biases1 = np.random.rand(128)

# layer 2: 128 -> 10

weights2 = np.random.rand(128, 10)
biases2 = np.random.rand(10)

learning_rate = 0.01


# run loop to train the model i.e get the weights and biases for the model

for epoch in range(10):  # number of epochs
    totalLoss = 0
    for i in range(len(images)):
        # forward pass
        image = images[i]
        label = labels[i]  

        z1 = np.dot(image, weights1) + biases1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, weights2) + biases2


        prob = softmax(z2); 


        loss = -np.log(prob[label]) 
        totalLoss += loss

        # backpropagation

        dz2 = prob.copy()
        dz2[label] -= 1

        dw2 = np.outer(a1, dz2)
        da1 = np.dot(weights2, dz2)

        dz1 = da1 * (z1 > 0)
        dw1 = np.outer(image, dz1)


        # update weights and biases
        weights2 -= learning_rate * dw2
        biases2 -= learning_rate * dz2

        weights1 -= learning_rate * dw1
        biases1 -= learning_rate * dz1

        print(f"Epoch {epoch + 1}, Image {i + 1}, Loss: {loss}")


np.savez(
    "mnist_model.npz",

    weights1=weights1,
    biases1=biases1,

    weights2=weights2,
    biases2=biases2
)
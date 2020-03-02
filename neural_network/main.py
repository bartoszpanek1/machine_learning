from neural_network.n_net import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from random import randrange
import warnings

warnings.filterwarnings("ignore")


a = loadmat('data.mat', squeeze_me=False)

print("------Neural network - Hand Written Digits Recognition------")
X, y = a["X"], a["y"]
data = np.asarray(X, dtype=np.float32)
labels = np.asarray(y, dtype=np.int32) % 10
m = data.shape[0]
b = loadmat('weights.mat')
Theta1, Theta2 = b["Theta1"], b["Theta2"]

#creating neural network object
nn = NeuralNetwork(3, [400, 25, 10], 0.1)

learning_rate = 0.9
iterations = 800
print("Learning on 5000 examples...")
costs = nn.gradient_descent(data, labels, learning_rate, iterations)
print("Learning finished")
plt.plot([i for i in range(1, iterations + 1)], costs)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost minimization - neural network')
plt.show()  # how gradient minimized the cost

print("Predicting 5000 examples..")
predictions = nn.predict(X)[0]
print("Finished predicting")
correct_counter = 0
for i in range(0, len(predictions)):
    if predictions[i] == labels[i]:
        correct_counter += 1

accuracy = round(correct_counter/m * 100,2)
print("Training set accuracy: " + str(accuracy) + "%")

input("Press any button...")
print("---------VISUALIZATION----------")

while True:

    random_choice = randrange(0, m)
    d = data[random_choice]

    plt.gray()
    plt.matshow(d.reshape((20,20), order="F"))
    plt.show()

    predicted = predictions[random_choice]
    print("Recognized as: "+ str(predicted))
    choice = input ("Press Enter to continue, write STOP to stop visualization: ")
    if choice.upper() == "STOP":
        break




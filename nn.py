"""
A simple neural network using the Iris dataset from UC Irvine, and pytorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    # Input layer (4 features of the flower)
    # -> Hidden Layer 1 - h1 (number of neurons)
    # -> Hidden Layer 2 - h2 (number of neurons)
    # -> Output (3 classes of Iris flowers)

    def __init__(self, input_features=4, h1=8, h2=9, output_features=3):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(input_features, h1)
        self.fully_connected_layer_2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, output_features)

    def move_forward(self, x):
        x = F.relu(self.fully_connected_layer_1(x))
        x = F.relu(self.fully_connected_layer_2(x))
        x = self.output_layer(x)
        return x

# Pick a manual seed for randomization
seed = torch.manual_seed(41)

model = Model()

import pandas
import matplotlib.pyplot as plt

# Load data
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
iris_df = pandas.read_csv(url)
iris_df['variety'] = iris_df['variety'].replace('Setosa', 0.0)
iris_df['variety'] = iris_df['variety'].replace('Versicolor', 1.0)
iris_df['variety'] = iris_df['variety'].replace('Virginica', 2.0)

"""
     sepal.length  sepal.width  petal.length  petal.width  variety
0             5.1          3.5           1.4          0.2      0.0
1             4.9          3.0           1.4          0.2      0.0
2             4.7          3.2           1.3          0.2      0.0
3             4.6          3.1           1.5          0.2      0.0
4             5.0          3.6           1.4          0.2      0.0
..            ...          ...           ...          ...      ...
145           6.7          3.0           5.2          2.3      2.0
146           6.3          2.5           5.0          1.9      2.0
147           6.5          3.0           5.2          2.0      2.0
148           6.2          3.4           5.4          2.3      2.0
149           5.9          3.0           5.1          1.8      2.0
"""

# Train, test, and split

# Drop the last column because the name of the variety is not a feature (we are trying to classify an iris based on features)
X = iris_df.drop('variety', axis=1)
y = iris_df['variety']

X = X.values
y = y.values

# Train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to FloatTensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert Y labels to LongTensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set the criterion of model to measure error (how far off the predictions are from the data)
criterion = nn.CrossEntropyLoss()

# Choose an optimizer and set learning rate
# If error doesn't go down after iterations (epochs) we can change the learning_rate
# The lower the learning_rate, the longer it will take to train the model

# model.parameters() are the properties of the Model class
learning_rate = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model!
# Epochs? (one run through all the data in our network)
epochs = 100
losses = []
for num in range(epochs):
    # Go forward and get a prediction
    y_prediction = model.move_forward(X_train)

    # Measure the loss/error, will likely be high at first
    loss = criterion(y_prediction, y_train)

    # Save losses for graphing them out in the future
    losses.append(loss.detach().numpy())

    # Print every 10 epochs
    if num % 10 == 0:
        print(f'Epoch {num}, loss {loss}')

    # Do some back propagation
    # Take the error rate of forward propagation and feed it bac
    # through the network to fine-tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss/Error')
plt.xlabel('Epoch')
# 2. Set the random seed
import random
random.seed(10)

# 3. Import the necessary libraries
import numpy as np  # This is used to generate the data
import re   # This is used to generate the data
import math # This is used to generate the data
import torch    # This is the main library
import torch.nn as nn   # This is used to create the model
import torch.optim as optim # This is used to optimize the model
import matplotlib  # This is used to plot the graph
import matplotlib.pyplot as plt # This is used to plot the graph

# 4. Define the function to generate the data
def generate_data(n):
    # Generate random data
    x = np.random.rand(n, 1)
    y = 5 * x + 3 + np.random.randn(n, 1) * 0.1
    return x, y

# 5. Define the function to create the model   
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
# 6. Define the function to train the model
def train_model(x, y, model, optimizer, criterion, epochs):
    for epoch in range(epochs):
        inputs = torch.from_numpy(x).float()
        labels = torch.from_numpy(y).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

# 7. Define the function to plot the graph
def plot_graph(x, y, model):
    plt.scatter(x, y)
    x = np.linspace(0, 1, 100)
    y = model(torch.from_numpy(x).float()).detach().numpy()
    plt.plot(x, y, 'r')
    plt.show()

# 8. Generate the data
n = 100
x, y = generate_data(n)

# 9. Create the model
model = LinearRegression()

# 10. Set the criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 11. Train the model
train_model(x, y, model, optimizer, criterion, 1000)

# 12. Plot the graph
plot_graph(x, y, model)

# 13. Save the model
torch.save(model.state_dict(), 'model.pth')



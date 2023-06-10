# Import the necessary libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('tesla_stock_data.csv')

# Extract the 'Close' prices
prices = data['Close'].values.reshape(-1, 1)

# Normalization
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices)

# Price Difference
data['PriceDiff'] = data['Close'].shift(-1) - data['Close']
print(data['PriceDiff'])

# Historical Close Prices
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title('Historical Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.show()

# Split the dataset
train_size = int(0.8 * len(prices))
train_data = prices[:train_size]
test_data = prices[train_size:]

# Prepare the data for the model:
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# the sequence length
seq_length = 10

# input sequences and labels
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Convert the data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the model

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Set the model hyperparameters

input_size = 1
hidden_size = 32
output_size = 1

# Create the LSTM model
model = LSTM(input_size, hidden_size, output_size)

# Training the model:
# Set the training parameters
num_epochs = 100
learning_rate = 0.001

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Evaluate the model

# Switch to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    train_predictions = model(X_train)
    test_predictions = model(X_test)

# Inverse transform the predictions to obtain actual prices

train_predictions = scaler.inverse_transform(train_predictions.detach().numpy())
test_predictions = scaler.inverse_transform(test_predictions.detach().numpy())

# Calculate the root mean squared error (RMSE)
train_rmse = np.sqrt(criterion(torch.from_numpy(train_predictions), y_train).item())
test_rmse = np.sqrt(criterion(torch.from_numpy(test_predictions), y_test).item())

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

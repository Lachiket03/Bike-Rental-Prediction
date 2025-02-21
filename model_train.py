import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib  # For saving the scaler and unique categories

# Load your dataset (adjust the path as needed)
data = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1')

# Preprocessing the data (handle categorical columns)
def preprocess_data(data):
    # Convert 'Date' column to datetime and extract useful features
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.month  # Extract month for cyclical encoding

    # Cyclical encoding for Hour and Month
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

    # Drop unused columns
    data.drop(columns=['Date', 'Hour', 'Month'], inplace=True)

    # One-hot encode categorical columns like 'Seasons', 'Holiday', and 'Functioning Day'
    one_hot_features = ['Seasons', 'Holiday', 'Functioning Day']
    data = pd.get_dummies(data, columns=one_hot_features, drop_first=True)

    # Save unique categories for one-hot encoding
    unique_categories = {feature: list(data.filter(like=feature).columns) for feature in one_hot_features}
    joblib.dump(unique_categories, 'unique_categories.pkl')

    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Save the scaler for consistent normalization
    joblib.dump(scaler, 'scaler.pkl')

    return scaled_data, data.columns

# Process data
X_processed, feature_columns = preprocess_data(data)

# Save the feature columns for reference
joblib.dump(feature_columns.tolist(), 'feature_columns.pkl')

# Select features (X) and target (y)
X = X_processed
y = data['Rented Bike Count']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Define the Neural Network model
class BikeRentalPredictor(nn.Module):
    def __init__(self, input_size):
        super(BikeRentalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model
model = BikeRentalPredictor(input_size=X_train.shape[1])  # input_size should match number of features
model.eval()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'bike_rental_model.pth')
print("Model saved successfully!")

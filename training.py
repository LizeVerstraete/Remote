import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        labels = [float(value) for value in self.data.iloc[idx, 1:].values]

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': labels}


# Define your transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of the dataset
dataset = CustomDataset(csv_file=r"C:\Users\Lize\Desktop\School\Master\thesis\dataset.csv", transform=transform)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create a multi-output regression model (linear regression in this case)
model = MultiOutputRegressor(LinearRegression())

# Convert PIL images to NumPy arrays and labels to NumPy arrays
X_train = np.array([np.array(sample['image']) for sample in train_dataset])
y_train = np.array([sample['label'] for sample in train_dataset])

X_test = np.array([np.array(sample['image']) for sample in test_dataset])
y_test = np.array([sample['label'] for sample in test_dataset])

# Flatten image arrays
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Fit the model on the training data
model.fit(X_train_flatten, y_train)

# Make predictions on the test data
predictions_test = model.predict(X_test_flatten)
predictions_train = model.predict(X_train_flatten)

# Evaluate the performance
mse_test = mean_squared_error(y_test, predictions_test)
mse_train = mean_squared_error(y_train,predictions_train)
print(f"Mean Squared Error test: {mse_test}")
print(f"Mean Squared Error train: {mse_train}")
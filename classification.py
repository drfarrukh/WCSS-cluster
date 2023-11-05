# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Make sure to import Image from PIL
import PIL
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# %%
import pathlib
import torch
from torchvision.datasets import ImageFolder

# Define the URL for the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Use pathlib to get the data directory
data_dir = pathlib.Path('./data')  # Define your data directory path

# Download and extract the dataset
if not data_dir.is_dir():
    import requests
    import tarfile
    from io import BytesIO
    
    response = requests.get(dataset_url)
    tarfile.open(fileobj=BytesIO(response.content), mode="r|gz").extractall(data_dir)

# Define the image folder using torchvision's ImageFolder
image_dataset = ImageFolder(data_dir)

# Get the total number of images
image_count = len(image_dataset)
print(image_count)

#%%

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the data directory path
data_dir = './data/flower_photos'  # Use the correct path to your dataset

# Define your batch size, image height, and image width
batch_size = 32
img_height = 180
img_width = 180

#%%
# Define data transformations (you can customize these as needed)
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Create the ImageFolder dataset
image_dataset = datasets.ImageFolder(data_dir, transform=transform)

# Calculate the sizes for training and validation datasets
dataset_size = len(image_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into training and validation
train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])


# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Access class names from the class_to_idx attribute of the dataset
class_names = image_dataset.classes
print(class_names)

# %%
from torch.utils.data import DataLoader

# Define your batch size
batch_size = 128

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)



#%%
import torch
import torch.nn as nn
from torchsummary import summary

# Define the number of classes
num_classes = len(class_names)

# Define the PyTorch model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()  # Pass the class name and self as arguments
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img_height // 8) * (img_width // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create an instance of your PyTorch model
model = CustomCNN()

model.to(device)


# Print the model summary
summary(model, (3, img_height, img_width))


#%%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the number of epochs and initialize lists to store metrics
epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%')


#%%
# Plot the accuracy and loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')
plt.grid()

plt.tight_layout()
plt.show()

#%%

#%%
import torchvision.transforms as transforms

# Define data augmentation transformations
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop((img_height, img_width), scale=(0.9, 1.1))
])

# %%
"""Visualize a few augmented examples by applying data augmentation to the same image several times:"""

import matplotlib.pyplot as plt

# Assuming train_loader is your DataLoader with data augmentation
for batch in train_loader:
    images, _ = batch  # Assuming you don't need labels for visualization
    plt.figure(figsize=(10, 10))
    
    for i in range(9):
        augmented_images = data_augmentation(images)  # Apply data augmentation
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].permute(1, 2, 0).numpy().astype("uint8"))
        plt.axis("off")

    plt.show()
    break  # Break after the first batch for visualization
#%%
import torch
import torch.nn as nn

# Define a custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(64 * (img_height // 8) * (img_width // 8), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create an instance of your PyTorch model
model = CustomCNN(num_classes)

model.to(device)

# Print the model summary
from torchsummary import summary
summary(model, input_size=(3, img_height, img_width))


#%%

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the number of epochs and initialize lists to store metrics
epochs = 15
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%')

# Plot the accuracy and loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')
plt.grid()

plt.tight_layout()
plt.show()

#%%
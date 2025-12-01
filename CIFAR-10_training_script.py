# === CDS521 Dissertation: Improved CIFAR-10 CNN Training Script (v10) ===

# === Step 1: Import All Necessary Libraries ===
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from google.colab import drive
import os

print("All libraries imported successfully.")

# === Step 2: Mount Google Drive ===
drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/My Drive/CDS521_Dissertation_v10/'
os.makedirs(DRIVE_PATH, exist_ok=True)
print(f"Files will be saved to: {DRIVE_PATH}")

# === Step 3: Set Up Device (GPU or CPU) ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# === Step 4: Load and Normalize CIFAR-10 Data (WITH AUGMENTATION) ===

# Training transforms with data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test transforms (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 32

DATA_PATH = DRIVE_PATH + 'data'
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("CIFAR-10 dataset loaded successfully with data augmentation.")

# === Step 5: Define CNN with Light Dropout ===
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.5 to 0.2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = Net()
net.to(device)
print("Model created with light dropout (p=0.2).")

# === Step 6: Define Loss Function and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
print("Using Adam optimizer with weight decay.")

# === Step 7: Train the Model (20 epochs) ===
print('Starting Training...')
num_epochs = 20  # Increased from 10 to 20

epoch_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    batch_count = 0
    
    net.train()  # Ensure model is in training mode
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        batch_count += 1
        
        if i % 1000 == 999:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0
    
    avg_epoch_loss = epoch_loss / batch_count
    epoch_losses.append(avg_epoch_loss)
    print(f'[Epoch {epoch + 1}] Average loss: {avg_epoch_loss:.3f}')

print('Finished Training')

# === Step 8: Save the Trained Model ===
MODEL_SAVE_PATH = DRIVE_PATH + 'cifar10_model_v10.pth'
torch.save(net.state_dict(), MODEL_SAVE_PATH)
print(f'Model saved to: {MODEL_SAVE_PATH}')

# === Step 9: Plot Training Loss Curve ===
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))

LOSS_PLOT_PATH = DRIVE_PATH + 'training_loss_v10.png'
plt.savefig(LOSS_PLOT_PATH)
print(f"Training loss plot saved to: {LOSS_PLOT_PATH}")
plt.show()

# === Step 10: Test the Model ===
print('Calculating accuracy on the 10000 test images...')
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

# === Step 11: Per-Class Accuracy ===
print('\nPer-class accuracy:')
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

for i in range(10):
    class_acc = 100 * class_correct[i] / class_total[i]
    print(f'Accuracy of {classes[i]:>6s}: {class_acc:.2f}%')

# === Step 12: Generate and Plot Confusion Matrix ===
print('\nGenerating confusion matrix...')
all_preds = []
all_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])

plt.figure(figsize=(12, 8))
sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for CIFAR-10 Test Set (Accuracy: {accuracy:.2f}%)')

PLOT_SAVE_PATH = DRIVE_PATH + 'confusion_matrix_v10.png'
plt.savefig(PLOT_SAVE_PATH)
print(f"Confusion matrix plot saved to: {PLOT_SAVE_PATH}")
plt.show()

# === Step 13: Print Summary ===
print('\n' + '='*60)
print('TRAINING SUMMARY')
print('='*60)
print(f'Architecture: 2 Conv layers (6, 16 filters) + 3 FC layers')
print(f'Regularization: Dropout (p=0.2) + Weight Decay (1e-4)')
print(f'Data Augmentation: RandomHorizontalFlip + RandomCrop')
print(f'Optimizer: Adam (lr=0.001)')
print(f'Loss Function: CrossEntropyLoss')
print(f'Batch Size: {batch_size}')
print(f'Epochs: {num_epochs}')
print(f'Final Test Accuracy: {accuracy:.2f}%')
print('='*60)

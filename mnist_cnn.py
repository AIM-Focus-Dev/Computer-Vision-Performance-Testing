import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import numpy as np

# mps_device configuration
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    print ("MPS mps_device not found.")
    mps_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
learning_rate = 0.005
batch_size = 64
hidden_size = 128
num_classes = 10
num_epochs = 100


# MNIST dataset

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# cnn model

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = CNN().to(mps_device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(mps_device)
        labels = labels.to(mps_device)
        
        outputs = model(images)
        cost = loss(outputs, labels)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), cost.item()))


# Test the model
model.eval()
for i, (images, labels) in enumerate(test_loader):
    images = images.to(mps_device)
    labels = labels.to(mps_device)
    
    # Get model predictions and probabilities
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)  # Predicted classes
    probabilities = torch.softmax(outputs, dim=1).cpu().detach().numpy()  # Probabilities for each class

    # Print predictions and ground truth
    print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(batch_size)))
    print('Ground truth: ', ' '.join('%5s' % labels[j].item() for j in range(batch_size)))

    # Calculate metrics
    accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
    recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    print('Accuracy: ', accuracy)
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1: ', f1)
    print("Classification Report: \n", classification_report(labels.cpu().numpy(), predicted.cpu().numpy()))

    # One-vs-All ROC Curve for a selected class (e.g., class '0')
    selected_class = 0  # You can choose other classes
    true_labels = (labels.cpu().numpy() == selected_class).astype(int)  # Binary ground truth
    predicted_probs = probabilities[:, selected_class]  # Probabilities for the selected class

    # Compute ROC Curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall Curve
    precision_pr, recall_pr, _ = precision_recall_curve(true_labels, predicted_probs)

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"Class {selected_class} (AUC = {roc_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title(f"ROC Curve for Class {selected_class}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall_pr, precision_pr, label=f"Class {selected_class}", color="green")
    plt.title(f"Precision-Recall Curve for Class {selected_class}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

    break 

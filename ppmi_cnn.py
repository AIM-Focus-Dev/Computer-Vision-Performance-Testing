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
num_classes = 12
num_epochs = 5

# PPMI dataset

train_dataset = torchvision.datasets.ImageFolder(root='./data/norm_ppmi_12class', transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(root='./data/norm_ppmi_12class', transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# cnn model
class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=1 ), # 256 after this layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 128 after
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1), # 126 after this layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 64 after
        self.fc = nn.Linear(128 * 61 * 61, num_classes)  # Flattened size * num_classes

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = CNN().to(mps_device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(mps_device)
        labels = labels.to(mps_device)
        # Forward pass
        outputs = model(images)
        loss_val = loss(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_val.item()}')


# test model
model.eval()
with torch.no_grad():  # Disable gradient computation
    total_accuracy, total_recall, total_precision, total_f1 = 0, 0, 0, 0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(mps_device), labels.to(mps_device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # Metrics
        accuracy = accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')

        total_accuracy += accuracy
        total_recall += recall
        total_precision += precision
        total_f1 += f1

        # Print metrics for each batch
        print(f"\nBatch {i+1}/{len(test_loader)} Metrics:")
        print(f"Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}")

        # Adjust predictions display for smaller batches
        print('Predicted: ', ' '.join(f'{predicted[j].item()}' for j in range(len(predicted))))
        print('Ground truth: ', ' '.join(f'{labels[j].item()}' for j in range(len(labels))))

        # Optional: Add ROC and Precision-Recall Curve plots
        if i == 0:  # Example: Plot for first batch only
            selected_class = 0
            true_labels = (labels.cpu().numpy() == selected_class).astype(int)
            predicted_probs = probabilities[:, selected_class]

            fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
            roc_auc = auc(fpr, tpr)
            precision_pr, recall_pr, _ = precision_recall_curve(true_labels, predicted_probs)

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, label=f"Class {selected_class} (AUC = {roc_auc:.2f})", color="blue")
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.title(f"ROC Curve for Class {selected_class}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid()
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(recall_pr, precision_pr, label=f"Class {selected_class}", color="green")
            plt.title(f"Precision-Recall Curve for Class {selected_class}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower left")
            plt.grid()
            plt.show()

# Print overall metrics
num_batches = len(test_loader)
print("\nOverall Test Metrics:")
print(f"Accuracy: {total_accuracy / num_batches:.2f}")
print(f"Recall: {total_recall / num_batches:.2f}")
print(f"Precision: {total_precision / num_batches:.2f}")
print(f"F1 Score: {total_f1 / num_batches:.2f}")


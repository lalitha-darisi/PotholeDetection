import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Define paths
pothole_directory = r'newdata\Dataset\Pothole'
normal_directory = r'newdata\Dataset\Normal'

flattened_images = []
labels = []

# Process pothole images
for image_file in os.listdir(pothole_directory):
    image_path = os.path.join(pothole_directory, image_file)
    image = Image.open(image_path).convert('L').resize((20, 20))
    flattened_images.append(np.array(image).flatten())
    labels.append(1)

# Process normal images
for image_file in os.listdir(normal_directory):
    image_path = os.path.join(normal_directory, image_file)
    image = Image.open(image_path).convert('L').resize((20, 20))
    flattened_images.append(np.array(image).flatten())
    labels.append(0)

# Convert to NumPy arrays
flattened_images = np.array(flattened_images, dtype=np.float32) / 255.0  # Normalize
labels = np.array(labels, dtype=np.float32)

# Convert to PyTorch tensors
X = torch.tensor(flattened_images)
y = torch.tensor(labels)

# Manually split data into training and testing sets
def train_test_split_manual(data, labels, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * (1 - test_ratio))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_ratio=0.2)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the network
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the network and track loss and accuracy
def train_model(model, train_loader, criterion, optimizer, epochs=30):
    model.train()
    train_losses = []
    accuracies = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Accuracy calculation
            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        epoch_accuracy = (correct_predictions / total_samples) * 100
        train_losses.append(epoch_loss / len(train_loader))
        accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return train_losses, accuracies

train_losses, accuracies = train_model(model, train_loader, criterion, optimizer, epochs=30)

# Save the model's state dictionary after training
torch.save(model.state_dict(), 'model2.pth')
print("Model saved successfully!")

# Evaluate the network
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.extend((outputs.squeeze() > 0.5).int().tolist())
            true_labels.extend(targets.int().tolist())
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    accuracy = np.mean(predictions == true_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return predictions, true_labels

predictions, true_labels = evaluate_model(model, test_loader)

# # Plotting the loss and accuracy graphs
# plt.figure(figsize=(14, 6))

# # Loss Plot
# plt.subplot(1, 2, 1)
# plt.plot(range(1, len(train_losses) + 1), train_losses, label="Loss", marker='o', color='red')
# plt.scatter(range(10, len(train_losses) + 1, 10), train_losses[9::10], color='blue')  # Mark every 10th epoch
# for i, val in enumerate(range(10, len(train_losses) + 1, 10)):
#     plt.text(val, train_losses[val - 1], f"{train_losses[val - 1]:.2f}", fontsize=8, ha='center')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss over Epochs")
# plt.legend()
# plt.grid(True)

# # Accuracy Plot
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(accuracies) + 1), accuracies, label="Accuracy", marker='o', color='green')
# plt.scatter(range(10, len(accuracies) + 1, 10), accuracies[9::10], color='orange')  # Mark every 10th epoch
# for i, val in enumerate(range(10, len(accuracies) + 1, 10)):
#     plt.text(val, accuracies[val - 1], f"{accuracies[val - 1]:.2f}%", fontsize=8, ha='center')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy over Epochs")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# Display images with their predicted and true labels
# plt.figure(figsize=(10, 5))
# for i, (image, pred, true_label) in enumerate(zip(X_test[:10], predictions[:10], y_test[:10])):  # Limit to 10 images
#     plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
#     plt.imshow(image.numpy().reshape(20, 20), cmap='gray')
#     label = 'Pothole' if pred == 1 else 'Non-Pothole'
#     true_label_text = 'Pothole' if true_label == 1 else 'Non-Pothole'
#     plt.title(f"Pred: {label}\nTrue: {true_label_text}")
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

# Load the model from the saved file
print("came here")
model_loaded = NeuralNetwork(input_size=X_train.shape[1])
model_loaded.load_state_dict(torch.load('model2.pth'))
model_loaded.eval()
print("Model loaded successfully!")


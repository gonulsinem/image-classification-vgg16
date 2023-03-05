import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the saved model
model = torch.load('output\saved_model_best_loss_0.00.pt')
model = model.to(device)

# Load the test dataset
test_dataset = ImageFolder(root='train_data\\test', transform=transform)

# Define the DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Test the model
test_loss = 0.0
test_correct = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()

test_loss /= len(test_dataset)
test_acc = test_correct / len(test_dataset)

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
with open('output\\test_results.txt', 'w') as f:
    f.write(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

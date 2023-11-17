import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torchvision.datasets as dset
from torch.utils.data import Dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CIFAR10TwoTransforms(Dataset):
    def __init__(self, root, train, transform1, transform2, selected_classes=None):
        self.original_dataset = dset.CIFAR10(root=root, train=train, download=True, transform=None)
        self.transform1 = transform1
        self.transform2 = transform2

        self.selected_classes = selected_classes
        if self.selected_classes is None:
            self.selected_classes = [0,1,2,3,4,5,6,7,8,9]
        # Filtering indices for selected classes
        self.filtered_indices = [i for i, (_, y) in enumerate(self.original_dataset) if y in self.selected_classes]
        self.class_names = [self.original_dataset.classes[i] for i in self.selected_classes]
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, label = self.original_dataset[original_idx]

        image_1 = self.transform1(image) if self.transform1 else image
        image_2 = self.transform2(image) if self.transform2 else image

        if self.transform2 is None:
            return image_1, label

        return image_1, label, image_2


# Neural Network Definition
class SimpleCNN(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        x = self.fc3(feat)
        if return_features:
            return x, feat
        return x

# Accuracy Calculation Function
def calculate_accuracy(loader, model, device='cpu'):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train(model, trainloader, testloader, epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_accuracy = calculate_accuracy(trainloader, model, device=device)
        test_accuracy = calculate_accuracy(testloader, model, device=device)
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}, Train accuracy: {train_accuracy}%, Test accuracy: {test_accuracy}%')

        # Save Model
        torch.save(model.state_dict(), f'cifar10_full_logs/model_epoch_{epoch+1}.pth')

    print('Finished Training')

def test(model, testloader, device='cpu'):
    # Load Model
    model.load_state_dict(torch.load('cifar10_full_logs/model_epoch_10.pth'))

    # Test Accuracy
    test_accuracy = calculate_accuracy(testloader, model, device=device)
    print(f'Test accuracy: {test_accuracy}%')

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data Preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Loading and Subsetting CIFAR-10 Dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # # Selecting classes 0, 1, and 2
    # train_indices = [i for i, (_, y) in enumerate(full_trainset) if y in [0, 1, 2]]
    # test_indices = [i for i, (_, y) in enumerate(full_testset) if y in [0, 1, 2]]

    # trainset = Subset(full_trainset, train_indices)
    # testset = Subset(full_testset, test_indices)
    
    trainset = CIFAR10TwoTransforms(root='./data', train=True, transform1=transform, transform2=None, selected_classes=None)
    testset = CIFAR10TwoTransforms(root='./data', train=False, transform1=transform, transform2=None, selected_classes=None)

    # DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Model, Loss Function, and Optimizer
    net = SimpleCNN()

    net = net.to(device)
    # Training and Testing
    train(net, trainloader, testloader, epochs=30, device=device)
    test(net, testloader, device=device)
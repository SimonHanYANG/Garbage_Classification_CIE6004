import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from dataloader import GarbageDataset
from net import Net
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter object
writer = SummaryWriter('runs/garbage_classification/')

transform = transforms.Compose([
    transforms.Resize((136, 136)),  # Resize to a slightly larger image
    transforms.RandomCrop((128, 128)),  # Random crop back to desired size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.RandomRotation(20),  # Random rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Random color jitter
    transforms.ToTensor(),
])

# Load Garbage dataset
train_dataset = GarbageDataset('garbage_classification/train.txt', transform=transform)
val_dataset = GarbageDataset('garbage_classification/val.txt', transform=transform)
test_dataset = GarbageDataset('garbage_classification/test.txt', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define train function
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.size())
        # print(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Write train loss to TensorBoard
        writer.add_scalar('Train Loss', loss.item(), epoch*len(train_loader) + batch_idx)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define test function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    # Write test loss and accuracy to TensorBoard
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', 100. * correct / len(test_loader.dataset), epoch)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Training/Testing settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Learning rate decay

epochs = 200
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, val_loader, criterion)
    scheduler.step()  # Update learning rate at the end of the epoch
    
# Close the SummaryWriter when you're done using it
writer.close()

# Test the model
# test(model, device, test_loader, criterion)


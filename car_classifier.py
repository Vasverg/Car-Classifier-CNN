import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchmetrics import Accuracy, Precision, Recall 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']

#Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.6),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(contrast=0.6),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(hue=0.2),]), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Data Loading
training_data = datasets.ImageFolder(r'C:\Users\billy\Desktop\Python Scripts\car_classifier\training', transform=train_transforms)
training_loader = DataLoader(training_data, batch_size=64, shuffle = True, num_workers = 0)

test_data = datasets.ImageFolder(r'C:\Users\billy\Desktop\Python Scripts\car_classifier\test', transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=64, shuffle = True, num_workers = 0)


def showimg(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#CNN
class CarClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d (3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d (32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d (64, 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d (128, 256, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d (2, 2)
        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear (256*14*14, 512)
        self.fc2 = nn.Linear (512, 7)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view (-1, 256*14*14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = CarClassifier().to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochsnum = 50
dataiter = iter(training_loader)
sample_images, sample_labels = next(dataiter)
showimg(torchvision.utils.make_grid(sample_images))

print(f"Initializing Training.")
for epoch in range(epochsnum):
    model.train()
    running_loss = 0.0
    for images, labels in training_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()
        running_loss += l.item()
    avg_loss =  running_loss / len(training_loader)
    print(f"Epoch [{epoch + 1}/{epochsnum}] Loss: {avg_loss:.4f}")

print("Model has been succesfully trained.")

acc = Accuracy(task = "multiclass", num_classes= 7).to(device)
pres = Precision(task = "multiclass", num_classes= 7, average = 'macro').to(device)
rec = Recall(task = "multiclass", num_classes= 7, average = 'macro').to(device)

print ("Initializing testing.")
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        max, preds = torch.max(outputs, 1)
        acc(preds, labels)
        pres(preds, labels)
        rec(preds, labels)

print(f"Test Accuracy: {acc.compute():.3f}")
print(f"Test Precision: {pres.compute():.3f}")
print(f"Test Recall: {rec.compute():.3f}")

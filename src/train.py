import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

from model import TrafficSignCNN

import matplotlib.pyplot as plt

from torchvision import transforms 

from PIL import Image
from torch.utils.data import Dataset

transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

class GTSRBTrainDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        for class_name in sorted(os.listdir(root_dir), key=int):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = int(class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".png"):
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = GTSRBTrainDataset(
    root_dir="data/GTSRB/Train",
    transform=transform
)

from torch.utils.data import DataLoader

train_loader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

images, labels=next(iter(train_loader))

# print(images.shape)
# print(labels.shape)

model=TrafficSignCNN()

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs=10

for epoch in range(num_epochs):

    model.train()

    running_loss=0.0
    
    for images, labels in train_loader:

        outputs=model(images)

        loss=criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    epoch_loss=running_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

import os

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/traffic_sign_cnn.pth"
)


        

        
        



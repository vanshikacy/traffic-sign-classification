import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import TrafficSignCNN

class GTSRBTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row=self.data.iloc[idx]
        
        img_path=os.path.join(
            self.root_dir,
            os.path.basename(row["Path"])
        )
        
        image=Image.open(img_path).convert("RGB")
        
        if self.transform:
            image=self.transform(image)
            
        label=int(row["ClassId"])
        
        return image, label 
        

transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

test_dataset=GTSRBTestDataset(
    csv_file="data/GTSRB/Test.csv",
    root_dir="data/GTSRB/Test",
    transform=transform
)

test_loader=DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False 
)

model=TrafficSignCNN()

model.load_state_dict(
    torch.load("models/traffic_sign_cnn.pth")
)

model.eval()

criterion=nn.CrossEntropyLoss()

running_loss=0.0
correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        outputs=model(images)

        loss=criterion(outputs, labels)
        running_loss+=loss.item()

        _, predicted=torch.max(outputs, dim=1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)

    test_loss=running_loss/len(test_loader)
    accuracy=100*correct/total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

        

        

        
        





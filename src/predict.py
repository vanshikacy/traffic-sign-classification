import torch
from PIL import Image
from torchvision import transforms 

from model import TrafficSignCNN

transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

model=TrafficSignCNN()

model.load_state_dict(
    torch.load("models/traffic_sign_cnn.pth")
)

model.eval()

image_path = "data/GTSRB/Test/00000.png"

image=Image.open(image_path).convert("RGB")

image=transform(image)
image=image.unsqueeze(0)

with torch.no_grad():
    outputs=model(image)
    _, prediction=torch.max(outputs, 1)

print(f"Predicted Class: {prediction.item()}")
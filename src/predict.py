import torch
from PIL import Image
from torchvision import transforms 

from src.model import TrafficSignCNN
#main.py is outside the src folder so when it runs this we need to start from root 

transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

model=TrafficSignCNN()

model.load_state_dict(
    torch.load("models/traffic_sign_cnn.pth")
)

model.eval()

CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# old way: hard coding
# image_path = "data/GTSRB/Test/00000.png"
# image=Image.open(image_path).convert("RGB")
# image=transform(image)
# image=image.unsqueeze(0)
# with torch.no_grad():
#     outputs=model(image)
#     _, prediction=torch.max(outputs, 1)
# print(f"Predicted Class: {prediction.item()}")

# new way: plug the image from user i/p into a func 
def predict_image(image): # api will handle image.open

    image=transform(image)
    image=image.unsqueeze(0)

    with torch.no_grad():
        outputs=model(image)
        _, prediction=torch.max(outputs, 1)

        predicted_class=prediction.item()

        probabilities=torch.softmax(outputs, dim=1)
        confidence=probabilities.max().item()

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4)
        }



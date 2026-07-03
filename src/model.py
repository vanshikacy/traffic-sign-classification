import torch.nn as nn

class TrafficSignCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3
        )

        self.relu=nn.ReLU()

        self.pool=nn.MaxPool2d(kernel_size=2)

        self.flatten=nn.Flatten()

        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.fc1=nn.Linear(
            in_features=2304,
            out_features=128
        )

        self.fc2=nn.Linear(
            in_features=128,
            out_features=43
        )

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        
        x=self.flatten(x)

        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x

# Used during development to determine the flattened feature size (2304).
# Uncomment if the architecture changes.

# if __name__=="__main__":
#     import torch

#     model=TrafficSignCNN()
#     dummy=torch.randn(1, 3, 32, 32)
#     output=model(dummy)
#     print(output.shape)

        
        
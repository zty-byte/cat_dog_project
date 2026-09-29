import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.classifier=nn.Sequential(
            nn.Linear(32*54*54,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
if __name__ == "__main__":

    model = CNN()

    x = torch.randn(32,3,224,224)

    output = model(x)

    print(output.shape)
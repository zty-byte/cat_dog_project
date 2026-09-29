import torch
import torch.nn as nn

from torch.optim import Adam 

from dataset import get_dataloader
from model import CNN

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

train_loader,val_loader=get_dataloader()
model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


num_epochs=10
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs}")
    running_loss=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    epoch_loss=running_loss/len(train_loader)
    print(f"Loss: {epoch_loss:.4f}")
    
    accuracy = validate(
        model,
        val_loader,
        device
    )
    print(f"Validation Accuracy: {accuracy:.4f}")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split,Subset

DATA_ROOT = "data/train"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

def get_dataloader(batch_size=32):

    train_dataset = datasets.ImageFolder(
        root=DATA_ROOT,
        transform=train_transform
)

    val_dataset = datasets.ImageFolder(
        root=DATA_ROOT,
        transform=val_transform
)

    dataset_size = len(train_dataset)

    indices = torch.randperm(dataset_size)

    train_indices = indices[:20000]
    val_indices = indices[20000:]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return train_loader,val_loader
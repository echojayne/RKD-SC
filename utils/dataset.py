from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

train_transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=transforms.functional.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=[-0.1,0.1]),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],[0.229, 0.224, 0.225]) ]
)

test_transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=transforms.functional.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073],[0.229, 0.224, 0.225])]
) 

def get_datasets(dataset_path, batch_size, dataset_name='cifar100'):
    if dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(dataset_path, True, transform=train_transform, download=False)
        test_dataset = datasets.CIFAR100(dataset_path, False, transform=test_transform, download=False)

    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(dataset_path, True, transform=train_transform, download=False)
        test_dataset = datasets.CIFAR10(dataset_path, False, transform=test_transform, download=False)

    else:
        raise ValueError('Invalid dataset name')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
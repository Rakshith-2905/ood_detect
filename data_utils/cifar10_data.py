import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import os
import argparse
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CIFAR10TwoTransforms(Dataset):
    def __init__(self, root, train, transform1, transform2, selected_classes=None):
        self.original_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True, transform=None)
        self.transform1 = transform1
        self.transform2 = transform2

        self.selected_classes = selected_classes
        if self.selected_classes is None:
            self.selected_classes = [0,1,2,3,4,5,6,7,8,9]
        # Filtering indices for selected classes
        self.filtered_indices = [i for i, (_, y) in enumerate(self.original_dataset) if y in self.selected_classes]
        self.class_names = [self.original_dataset.classes[i] for i in self.selected_classes]
    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, label = self.original_dataset[original_idx]

        image_1 = self.transform1(image) if self.transform1 else image
        image_2 = self.transform2(image) if self.transform2 else image

        if self.transform2 is None:
            return image_1, label

        return image_1, label, image_2

class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, corruption='gaussian_blur', transform=None,clip_transform=None,level=0):
        numpy_path = f'data/CIFAR-10-C/{corruption}.npy'
        t = 10000
        self.transform = transform
        self.clip_transform = clip_transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:]
        self.data = self.data_[:t,:,:,:]
        self.targets_ = np.load('data/CIFAR-10-C/labels.npy')
        self.targets = self.targets_[:t]
        self.np_PIL = transforms.Compose([transforms.ToPILImage()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image_ = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image_)
            image_to_clip = self.clip_transform(self.np_PIL(image_))
        targets = self.targets[idx]
        return image, targets, image_to_clip

def get_CIFAR10_dataloader(batch_size=512, data_dir='./data', selected_classes=None,    
                        train_transform=None, test_transform=None, clip_transform=None, subsample_trainset=True, return_dataset=False):

    # TODO:Change the mean and std to the ones for CIFAR-10-C #Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
    if train_transform is None:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    if test_transform is None:
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    temp_train_dataset = CIFAR10TwoTransforms(root=data_dir, train=True, transform1=test_transform, transform2=clip_transform, 
                                              selected_classes=selected_classes)
    test_dataset = CIFAR10TwoTransforms(root=data_dir, train=False, transform1=test_transform, transform2=clip_transform, 
                                        selected_classes=selected_classes)

    # Split trainset into train, val
    val_size = int(0.20 * len(temp_train_dataset))
    train_size = len(temp_train_dataset) - val_size
    train_dataset, temp_valset = torch.utils.data.random_split(temp_train_dataset, [train_size, val_size], 
                                                               generator=torch.Generator().manual_seed(42))
    
    # random subsample trainset to 20% of original size
    if subsample_trainset:
        train_size = int(0.20 * len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size], 
                                                         generator=torch.Generator().manual_seed(42))

    # Split the valset into val and failure
    failure_size = int(0.50 * len(temp_valset))
    val_size = len(temp_valset) - failure_size
    val_dataset, failure_dataset = torch.utils.data.random_split(temp_valset, [val_size, failure_size], 
                                                                 generator=torch.Generator().manual_seed(42))

    if return_dataset:
        return train_dataset, val_dataset, test_dataset, failure_dataset, temp_train_dataset.class_names

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    failure_loader = DataLoader(failure_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'failure': failure_loader,
        'test': test_loader
    }

    return loaders, temp_train_dataset.class_names


def get_CIFAR10C_dataloader(batch_size=512, data_dir='./data', corruption='gaussian_blur', severity=3,
                        train_transform=None, test_transform=None, clip_transform=None, return_dataset=False):

    # TODO:Change the mean and std to the ones for CIFAR-10-C #Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
    if train_transform is None:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    if test_transform is None:
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])


    train_dataset = CIFAR10C(corruption=corruption, transform=train_transform,
                             clip_transform=clip_transform,level=severity)
    val_dataset = train_dataset
    failure_dataset = train_dataset
    test_dataset = train_dataset

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']                                           

    if return_dataset:
        return train_dataset, val_dataset, test_dataset, failure_dataset, class_names

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    failure_loader = DataLoader(failure_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'failure': failure_loader,
        'test': test_loader
    }

    return loaders, class_names




if __name__ == "__main__":

    loaders, class_names = get_CIFAR10_dataloader(batch_size=512, data_dir='./data', selected_classes=None, 
                                                    train_transform=None, test_transform=None, subsample_trainset=False)                 
    print(class_names)
    print(len(loaders['train'].dataset))
    print(len(loaders['val'].dataset))
    print(len(loaders['failure'].dataset))
    print(len(loaders['test'].dataset))

    for i, (images, labels) in enumerate(loaders['train']):
        print(images.shape)
        print(labels.shape)
        break
    
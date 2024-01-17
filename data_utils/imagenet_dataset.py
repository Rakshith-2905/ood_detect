import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder

import os
import argparse
import numpy as np

from data_utils.imagenet_class_index import IN_CLASS_MAPPING, all_wnids, imagenet_r_wnids, imagenet_a_wnids

class ImageNetTwoTransforms(ImageFolder):
    def __init__(self, root, split, primary_transform, secondary_transform=None, data_type='imagenet', **kwargs):
        super().__init__(root, **kwargs)
        self.primary_transform = primary_transform
        self.secondary_transform = secondary_transform

        # Get the class names from the IN_CLASS_MAPPING
        class_name_to_idx = {v[-1]: k for k, v in IN_CLASS_MAPPING.items()}
        wnids_to_idx = {v[0]: k for k, v in IN_CLASS_MAPPING.items()}
        wnids_to_class_names = {v[0]: v[-1] for v in IN_CLASS_MAPPING.values()}
        class_names_to_wnids = {v[-1]: v[0] for v in IN_CLASS_MAPPING.values()}

        self.class_names = list(class_name_to_idx.keys())

        if data_type == 'imagenet':
            self.wnids = all_wnids
        elif data_type == 'imagenet_r':
            self.wnids = imagenet_r_wnids
            self.class_names = [wnids_to_class_names[wnid] for wnid in self.wnids]            
        elif data_type == 'imagenet_a':
            self.wnids = imagenet_a_wnids
            self.class_names = [wnids_to_class_names[wnid] for wnid in self.wnids]
        else:
            raise ValueError(f'Unknown data_type: {data_type}')
        self.split = split


    def __getitem__(self, index):
        image, label = super(ImageNetTwoTransforms, self).__getitem__(index)
        
        primary_image = self.primary_transform(image) if self.primary_transform else image
        secondary_image = self.secondary_transform(image) if self.secondary_transform else image
        if self.secondary_transform is None:
            return primary_image, label
        return primary_image, label, secondary_image
    
def get_imagenet_loaders(batch_size=512, data_dir='./data',    
                        train_transform=None, test_transform=None, clip_transform=None, 
                        data_type='imagenet', subsample_trainset=True, return_dataset=False):
    
    if data_type == 'imagenet':
        train_data_dir = os.path.join(data_dir, 'imagenet-train', 'train')
        val_data_dir = os.path.join(data_dir, 'imagenet-train', 'val')
    else:
        train_data_dir = os.path.join(data_dir, data_type)
        val_data_dir = os.path.join(data_dir, data_type)

    temp_train_dataset = ImageNetTwoTransforms(root=train_data_dir, primary_transform=train_transform, 
                                               secondary_transform=clip_transform, data_type=data_type)

    if data_type == 'imagenet':
        test_dataset = ImageNetTwoTransforms(root=val_data_dir, primary_transform=test_transform, 
                                            secondary_transform=clip_transform, data_type=data_type)

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
    else:
        train_dataset = temp_train_dataset
        val_dataset = temp_train_dataset
        failure_dataset = temp_train_dataset
        test_dataset = temp_train_dataset

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

if __name__ == "__main__":

    loaders, class_names = get_imagenet_loaders(batch_size=512, data_dir='./data',
                                                train_transform=None, test_transform=None, clip_transform=None,
                                                data_type='imagenet', subsample_trainset=False, return_dataset=False)
    print(len(loaders['train'].dataset))
    print(len(loaders['val'].dataset))
    print(len(loaders['failure'].dataset))
    print(len(loaders['test'].dataset))
    print(len(class_names))
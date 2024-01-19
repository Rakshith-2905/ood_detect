import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import CIFAR100
import numpy as np
import os
from PIL import Image
from torchvision import transforms

import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR100C(Dataset):
    """ Custom Dataset for CIFAR100-C with specific corruption type """
    def __init__(self, data_dir, corruption_name, transform=None):
        self.data_dir = data_dir
        self.corruption_name = corruption_name
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        np_labels = np.load(os.path.join(self.data_dir, "labels.npy"))
        corruption_file = f"{self.corruption_name}.npy"
        np_images = np.load(os.path.join(self.data_dir, corruption_file))
        return np_images, np_labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = Image.fromarray(image)
        return self.transform(image) if self.transform else image, label

class CIFAR100TwoTransforms(Dataset):
    def __init__(self, root, train=True, transform1=None, transform2=None, selected_classes=None, retain_orig_ids=False):
        self.original_dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform1 = transform1
        self.transform2 = transform2

        self.class_names = self.original_dataset.classes

        self.targets = self.original_dataset.targets
        self.data = self.original_dataset.data
        if selected_classes is not None:
            # Create a mask for all the targets that are in the desired selected_classes
            mask = [t in selected_classes for t in self.targets]
            
            # Filter data and targets according to the mask
            self.data = self.data[mask]
            self.targets = [self.targets[idx] for idx, m in enumerate(mask) if m]
            
            # If we are not retaining the original class indices, remap them
            if not retain_orig_ids:
                # Create a mapping for the selected indices
                self.new_targets_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(selected_classes))}
                # Remap the targets
                self.targets = [self.new_targets_map[target] for target in self.targets]

                self.class_names = [self.class_names[idx] for idx in sorted(selected_classes)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        image_2 = image.copy()
        image_1 = self.transform1(image) if self.transform1 else image
        image_2 = self.transform2(image) if self.transform2 else image_2

        if self.transform2 is None:
            return image_1, label

        return image_1, label, image_2

def get_CIFAR100_dataloader(batch_size=512, data_dir='./data', selected_classes=None, retain_orig_ids=False,    
                        train_transform=None, test_transform=None, clip_transform=None, 
                        subsample_trainset=False, return_dataset=False):
    """
    Returns the dataloaders for CIFAR-10 dataset
    Args:
        batch_size: batch size for the dataloaders
        data_dir: directory to store the data
        selected_classes: list of classes to select from the dataset
        retain_orig_ids: whether to retain the original class indices in the filtered dataset or not
        train_transform: transform to apply to the trainset
        test_transform: transform to apply to the testset
        clip_transform: transform to apply to the clipset
        subsample_trainset: whether to subsample the trainset to 20% of the original size
        return_dataset: whether to return the dataset objects instead of the dataloaders
    """

    # TODO:Change the mean and std to the ones for CIFAR-10-C #Mean : [0.491 0.482 0.446]   STD: [0.247 0.243 0.261]
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy images to PIL Image format
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
                transforms.ToPILImage(),  # Convert NumPy images to PIL Image format
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            ])

    temp_train_dataset = CIFAR100TwoTransforms(root=data_dir, train=True, transform1=train_transform, transform2=clip_transform, 
                                              selected_classes=selected_classes, retain_orig_ids=retain_orig_ids)
    test_selected_classes = None
    # if selected_classes is not None:
    #     # Get the numbers that are missing from the selected classes in a range of 0-99
    #     test_selected_classes = list(set(list(range(100))) ^ set(selected_classes))


    test_dataset = CIFAR100TwoTransforms(root=data_dir, train=False, transform1=test_transform, transform2=clip_transform, 
                                        selected_classes=test_selected_classes, retain_orig_ids=retain_orig_ids)

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

def get_CIFAR100C_dataloader():
    pass

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='CIFAR100-loader')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_classes_train', type=int, default=90, help='number of classes in train set')
    parser.add_argument('--retain_orig_ids', action='store_true', help='retain the original class indices in the filtered dataset')
    args = parser.parse_args()

    # Sorting the classes
    np.random.seed(args.seed)
    select_indices = sorted(np.random.choice(list(range(100)), args.num_classes_train, replace=False))
    remaining_classes = list(set(list(range(100))) ^ set(select_indices))

    train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
        ])
    datasets = get_CIFAR100_dataloader(batch_size=args.batch_size, data_dir='data', 
                            selected_classes=None, retain_orig_ids=args.retain_orig_ids,
                            train_transform=None, test_transform=None, clip_transform=None,
                            subsample_trainset=False, return_dataset=True)
    
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = datasets

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(len(failure_dataset))
    print(class_names, len(class_names))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)

        if i == 10:
            break

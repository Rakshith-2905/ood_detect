import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision.datasets import CIFAR100
import numpy as np
import os
from PIL import Image
from torchvision import transforms

import argparse


class CIFAR100CDataset(Dataset):
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

class CIFAR100Filtered(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, select_indices=None, retain_orig_ids=False):
        super(CIFAR100Filtered, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        if select_indices is not None:
            # Create a mask for all the targets that are in the desired select_indices
            mask = [t in select_indices for t in self.targets]
            
            # Filter data and targets according to the mask
            self.data = self.data[mask]
            self.targets = [self.targets[idx] for idx, m in enumerate(mask) if m]
            
            # If we are not retaining the original class indices, remap them
            if not retain_orig_ids:
                # Create a mapping for the selected indices
                self.new_targets_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(select_indices))}
                # Remap the targets
                self.targets = [self.new_targets_map[target] for target in self.targets]

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


def get_CIFAR100_loaders(batch_size=512,train_shuffle=True, data_dir='./data', select_indices=None, retain_orig_ids=False,    
                        train_transform=None, test_transform=None, return_dataset=False):
    
    if train_transform is None:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
    if test_transform is None:
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    train_dataset = CIFAR100Filtered(root=data_dir, train=True, transform=train_transform, download=True, 
                                    select_indices=select_indices, retain_orig_ids=retain_orig_ids)
    test_dataset = CIFAR100Filtered(root=data_dir, train=False, transform=test_transform, download=True, 
                                    select_indices=select_indices, retain_orig_ids=retain_orig_ids)

    if return_dataset:
        return train_dataset, test_dataset, train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    loaders = {'train': train_loader, 'test': test_loader}
    
    # Get the class names
    class_names = train_dataset.classes
    return loaders, class_names


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

    # Get the loaders and class names
    loaders, class_names = get_CIFAR100_loaders(batch_size=args.batch_size, train_shuffle=True, data_dir='./data', \
                            select_indices=select_indices, retain_orig_ids=args.retain_orig_ids)

    print(f"Train classes: {select_indices}")
    print(f"Remaining classes: {remaining_classes}")

    # Print the number of images in train and test sets
    print(f"Number of train images: {len(loaders['train'].dataset)}")
    print(f"Number of test images: {len(loaders['test'].dataset)}")

    # Iterate over the train loader and accumate the labels in a set
    unique_train_labels = set()
    for images, labels in loaders['test']:
        unique_train_labels.update(labels.tolist())
    print(f"Unique Class IDs: {unique_train_labels}")

    print(f"\nClass names: {class_names}")


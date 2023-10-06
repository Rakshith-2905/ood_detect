import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

import torch
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None, class_mapping=None):
        self.subset = subset
        self.transform = transform
        self.class_mapping = class_mapping  # Added class mapping
    def __getitem__(self, index):
        x, y, z = self.subset[index]
        if self.transform:
            x = self.transform(x)
        # Remap y based on the class mapping
        if self.class_mapping:
            y = self.class_mapping[y.item()]
            y = torch.tensor(y)
        return x, y, z

    def __len__(self):
        return len(self.subset)
    
class WildsDataLoader:
    def __init__(self, dataset_name, split, image_size, batch_size, class_percentage=1.0, seed=42, selected_classes=None, use_train_classes=False):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_percentage = class_percentage
        self.seed = seed
        self.use_train_classes = use_train_classes
        self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
        self.data = None
        self.dataloader = None
        self.selected_classes = selected_classes

    def _get_filtered_indices(self, dataset):
        np.random.seed(self.seed)  # Setting seed for reproducibility
        unique_classes = np.unique(dataset.y_array)
        
        if self.selected_classes is None:
            num_selected_classes = int(len(unique_classes) * self.class_percentage)
            self.selected_classes = np.random.choice(unique_classes, num_selected_classes, replace=False)
        else:
            if not self.use_train_classes and self.split != 'train':
                remaining_classes = np.setdiff1d(unique_classes, self.selected_classes)
                num_required = int(len(unique_classes) * self.class_percentage) - len(self.selected_classes)
                additional_classes = np.random.choice(remaining_classes, num_required, replace=False)
                self.selected_classes = np.concatenate([self.selected_classes, additional_classes])

        # Sort the selected_classes array
        self.selected_classes = np.sort(self.selected_classes)
        # Create a mapping from original class label to new label
        self.class_mapping = {original: idx for idx, original in enumerate(self.selected_classes)}
        return np.where(np.isin(dataset.y_array, self.selected_classes))[0]


    def load_data(self):
        dataset_args = {
            'iwildcam': {'download': True, 'transform': self.transform},
            'domainnet': {'source_domain':'real', 'target_domain':'real'},
            # Add other datasets with their specific args here
            # 'other_dataset': {'arg1': value1, 'arg2': value2, ...}
        }

        # Load the full dataset with dataset-specific arguments
        dataset = get_dataset(dataset=self.dataset_name, **dataset_args.get(self.dataset_name, {}))

        # Filter by class percentage if less than 100%
        if self.class_percentage < 1.0:
            filtered_indices = self._get_filtered_indices(dataset)
            subset = Subset(dataset, filtered_indices)
            self.data = TransformedSubset(subset, transform=self.transform, class_mapping=self.class_mapping)
        else:
            unique_classes = np.unique(dataset.y_array)
            self.selected_classes = unique_classes
            self.data = dataset.get_subset(self.split, transform=self.transform)

        # Prepare the standard data loader
        if self.split == 'train':
            if self.class_percentage < 1.0:
                self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
            else:
                self.dataloader = get_train_loader('standard', self.data, batch_size=self.batch_size)
        else:
            if self.class_percentage < 1.0:
                # Using PyTorch's DataLoader with the collate function from the original dataset
                self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=False)
            else:
                self.dataloader = get_eval_loader('standard', self.data, batch_size=self.batch_size)
        return self.dataloader

    def display_details(self, show_images=False):
        print(f"Dataset Name: {self.dataset_name}")
        print(f"Split: {self.split}")
        print(f"Number of samples: {len(self.data)}")
        print(f"Image size: {self.image_size}")

        # Class-level statistics

        print("\nClass Level Statistics:")
        print(f"Class Percentage: {self.class_percentage}")
        if self.class_percentage < 1.0:
            print(f"Number of selected classes: {len(self.selected_classes)}")
        else:
            label_list = list(self.data.dataset.y_array.numpy())
            label_counter = Counter(label_list)
            print(f"Number of classes: {len(label_counter)}")

        # for class_id, count in label_counter.items():
        #     print(f"Class {class_id}: {count} samples")

        if show_images:
            fig, axs = plt.subplots(10, 10, figsize=(12, 12))
            for i, (images, labels, _) in enumerate(self.dataloader):
                if i == 10:
                    break
                for j in range(10):
                    axs[i, j].imshow(np.transpose(images[j].numpy(), (1, 2, 0)))
                    axs[i, j].set_title(f"Label: {labels[j].item()}")
                    axs[i, j].axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WILDS Dataset Loader Class')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'val'], required=True, help='Dataset split to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=1, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    data_loader = WildsDataLoader(args.dataset, args.split, args.image_size, args.batch_size, args.class_percentage, args.seed)
    data_loader.load_data()
    data_loader.display_details(show_images=True)

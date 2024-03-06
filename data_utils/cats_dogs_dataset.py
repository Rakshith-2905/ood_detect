import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision.datasets import ImageFolder

import os
import argparse
import numpy as np
import random

class CatsDogsTwoTransforms(ImageFolder):
    def __init__(self, root, split, transform1=None, transform2=None, imbalance_ratio=1):
        # Initialize the parent ImageFolder class
        super().__init__(os.path.join(root, split))

        self.transform1 = transform1
        self.transform2 = transform2
        self.imbalance_ratio = imbalance_ratio

        self.class_names = self.classes
        # Apply class imbalance if imbalance_ratio is different than 1
        if imbalance_ratio != 1:
            self._apply_class_imbalance()

    def _apply_class_imbalance(self):
        # Separate cats and dogs images
        cats = [item for item in self.imgs if 'cat' in item[0].lower()]
        dogs = [item for item in self.imgs if 'dog' in item[0].lower()]

        # Adjust the number of cat images based on the imbalance ratio
        reduced_cats_count = int(len(dogs) * self.imbalance_ratio)
        if reduced_cats_count < len(cats):  # Only reduce if it leads to fewer cats than currently exist
            # Use random seed generator to ensure reproducibility
            random.seed(42)
            cats = random.sample(cats, reduced_cats_count)

        # Print the number of images for each class
        print(f"Class imbalance applied on : {len(cats)} cats, {len(dogs)} dogs")

        # Combine the adjusted classes back into the dataset
        self.imgs = cats + dogs
        self.samples = self.imgs  # Update the samples attribute as well

    def __getitem__(self, index):
        # Override the method to apply different transforms
        path, target = self.imgs[index]
        img = self.loader(path)

        primary_image = self.transform1(img) if self.transform1 else img
        secondary_image = self.transform2(img) if self.transform2 else img
        if self.transform2 is None:
            return primary_image, target
        return primary_image, target, secondary_image
    
def get_cats_dogs_loaders(batch_size=512, data_dir='./data',    
                        train_transform=None, test_transform=None, clip_transform=None, 
                        return_dataset=False):
    
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    data_dir = os.path.join(data_dir, 'cats_dogs', 'PetImages')

    temp_train_dataset = CatsDogsTwoTransforms(root=data_dir, split='train', transform1=train_transform,
                                            transform2=clip_transform, imbalance_ratio=0.3)
    test_dataset = CatsDogsTwoTransforms(root=data_dir, split='test', transform1=test_transform,
                                            transform2=clip_transform, imbalance_ratio=1)
    
    # Split trainset into train, val
    val_size = int(0.20 * len(temp_train_dataset))
    train_size = len(temp_train_dataset) - val_size
    train_dataset, temp_valset = torch.utils.data.random_split(temp_train_dataset, [train_size, val_size], 
                                                               generator=torch.Generator().manual_seed(42))

    val_dataset = temp_valset
    failure_dataset = temp_valset
    # # Split the valset into val and failure
    # failure_size = int(0.50 * len(temp_valset))
    # val_size = len(temp_valset) - failure_size
    # val_dataset, failure_dataset = torch.utils.data.random_split(temp_valset, [val_size, failure_size], 
    #                                                             generator=torch.Generator().manual_seed(42))
    
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

def split_dataset():

    # Skip this step if the dataset is already split
    if os.path.exists(os.path.join('data', 'cats_dogs', 'PetImages', 'train')):
        return

    # Read the images in cat and dog folders and create test split

    cat_folder = os.path.join('data', 'cats_dogs', 'PetImages', 'Cat')
    dog_folder = os.path.join('data', 'cats_dogs', 'PetImages', 'Dog')

    cat_images = [os.path.join(cat_folder, img) for img in os.listdir(cat_folder)]
    dog_images = [os.path.join(dog_folder, img) for img in os.listdir(dog_folder)]

    # Split the images into train and test
    random.seed(42)
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    # 80% train, 20% test
    train_cat_images = cat_images[:int(0.8*len(cat_images))]
    test_cat_images = cat_images[int(0.8*len(cat_images)):]
    train_dog_images = dog_images[:int(0.8*len(dog_images))]
    test_dog_images = dog_images[int(0.8*len(dog_images)):]

    # Create the test folder
    test_folder = os.path.join('data', 'cats_dogs', 'PetImages', 'test')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'Dog'), exist_ok=True)

    # Move the test images to the test folder
    for img in test_cat_images:
        os.rename(img, os.path.join(test_folder, 'Cat', os.path.basename(img)))
    for img in test_dog_images:
        os.rename(img, os.path.join(test_folder, 'Dog', os.path.basename(img)))

    # Create the train folder
    train_folder = os.path.join('data', 'cats_dogs', 'PetImages', 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'Dog'), exist_ok=True)

    # Move the train images to the train folder
    for img in train_cat_images:
        os.rename(img, os.path.join(train_folder, 'Cat', os.path.basename(img)))
    for img in train_dog_images:
        os.rename(img, os.path.join(train_folder, 'Dog', os.path.basename(img)))


if __name__ == "__main__":


    # Split the dataset into train and test
    split_dataset()

    loaders, class_names = get_cat_dog_loaders(return_dataset=False)
    print(len(loaders['train'].dataset))
    print(len(loaders['val'].dataset))
    print(len(loaders['failure'].dataset))
    print(len(loaders['test'].dataset))
    print(len(class_names))
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

import numpy as np
import seaborn as sns

random.seed(0)
np.random.seed(0)

class FilteredCelebADataset(Dataset):
    def __init__(self, root_dir, attr_path='data/CelebA/CelebA/Anno/list_attr_celeba.txt',
                 partition_path='data/CelebA/CelebA/Eval/list_eval_partition.txt', 
                 landmark_path='data/CelebA/CelebA/Anno/list_landmarks_align_celeba.txt',
                 transform=None, transform1=None, split='tr', class_attr=None, num_images=30000,
                 imbalance_attr=None, imbalance_percent=None, ignore_attrs=None,
                 mask=False, mask_region=None):
        
        """
            root_dir: str
                Path to the directory containing the image files.
            
            attr_path: str, optional
                Path to the file containing the attributes for each image. The default path is set.
                
            partition_path: str, optional
                Path to the file containing the partition information for the dataset. The default path is set.
            
            landmark_path: str, optional
                Path to the file containing the landmark information for each image. The default path is set.
                
            transform: torchvision.transforms, optional
                Any transformations to apply to the images. Default is None.
                
            split: str, optional
                The partition of the dataset to use. {tr, va, te}. Default is tr.
            
            class_attr: str, optional
                The attribute to use for class labels. Default is None.
            
            num_images: int, optional
                The number of images to include in the dataset. Default is 30000.
            
            imbalance_attr: list of str, optional
                A list of attributes to create an imbalanced dataset. Default is None.
                
            imbalance_percent: list of float, optional
                A list of percentages to create an imbalanced dataset. Each element of the list corresponds
                to an attribute in imbalance_attr. Default is None.
                
            ignore_attrs: list of str, optional
                A list of attributes to ignore when creating the dataset. Default is None.
                
            mask: bool, optional
                Whether to apply a mask to the images. Default is False.
                
            mask_region: list of str, optional
                A list of regions to mask if mask is True. Each element of the list should correspond
                to a column name (minus '_x' or '_y') in the landmark file. Default is None.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.transform1 = transform1
        
        self.class_attr = class_attr
        self.imbalance_attr = imbalance_attr
        self.num_images = num_images
        self.imbalance_percent = imbalance_percent
        self.ignore_attrs = ignore_attrs if ignore_attrs is not None else []

        self.mask = mask
        self.mask_region = mask_region if mask_region is not None else []
        
        if split == 'tr':
            partition = 0
        elif split == 'va':
            partition = 1
        elif split == 'te':
            partition = 2

        self.partition = partition
        # load attributes
        self.attr_df = pd.read_csv(attr_path, delim_whitespace=True, header=1)
        self.attr_df = self.attr_df.replace(-1, 0) # replace -1 with 0

        # load partition
        self.partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None, names=['image_id', 'partition'])

        # filter based on partition
        self.images_df = self.partition_df[self.partition_df['partition'] == self.partition]

        # Reset the index of df1
        self.attr_df = self.attr_df.reset_index().rename(columns={'index': 'image_id'})
        # Merge df1 and df2
        self.attr_df = pd.merge(self.attr_df, self.images_df, on='image_id', how='inner')

        # load landmarks
        self.landmark_df = pd.read_csv(landmark_path, delim_whitespace=True, header=1)

        self.landmark_df = self.landmark_df.reset_index().rename(columns={'index': 'image_id'})
        self.landmark_df = pd.merge(self.landmark_df, self.images_df, on='image_id', how='inner')

        print(f"\nNumber of images in partition '{split}': {len(self.attr_df)}\n")


        self.generate_dataset()

    def generate_dataset(self):
        total_per_class = self.num_images // 2  # number of images per class

        dfs = []
        for class_value in [0, 1]:

            if class_value == 0:
                class_name = 'Not ' + self.class_attr
            else:
                class_name = self.class_attr

            class_df = self.attr_df[self.attr_df[self.class_attr] == class_value]
            
            
            # ignore certain attributes
            for attr in self.ignore_attrs:
                class_df = class_df[class_df[attr] == 0]

            # If imbalance_attr is empty, just sample images equally for each class
            if not self.imbalance_attr:
                df_sample = class_df.sample(total_per_class, replace=False)
                dfs.append(df_sample)
                print(f"\nClass '{class_name}': Sampled {len(df_sample)} images (No imbalance attributes).\n")

            elif len(self.imbalance_attr) == 1:
                # Special case: Only one attribute in imbalance_attr
                attr = self.imbalance_attr[0]
                percent = self.imbalance_percent[class_value][0]
                
                ################# Sampling for when att is True #################
                # Determine sample size
                attr_sample_size = int(total_per_class * percent / 100)
                attr_available = len(class_df[class_df[attr] == 1])

                self.print_sample_info(class_name, attr, True, attr_sample_size, attr_available)
                
                attr_sample_size = min(attr_sample_size, attr_available)

                # Sample with attribute true
                attr_df = class_df[class_df[attr] == 1].sample(attr_sample_size, replace=False)
                dfs.append(attr_df)

                ################# Sampling for when att is False #################
                attr_false_sample_size = total_per_class - attr_sample_size
                attr_false_available = len(class_df[class_df[attr] == 0])

                    
                self.print_sample_info(class_name, attr, False, attr_false_sample_size, attr_false_available)
                attr_false_sample_size = min(attr_false_sample_size, attr_false_available)

                attr_false_df = class_df[class_df[attr] == 0].sample(attr_false_sample_size, replace=False)
                dfs.append(attr_false_df)
            
            else:
                # Multiple attributes in imbalance_attr
                for attr, percent in zip(self.imbalance_attr, self.imbalance_percent[class_value]):
                    
                    # Determine sample size
                    attr_sample_size = int(total_per_class * percent / 100)
                    attr_available = len(class_df[class_df[attr] == 1])
                    self.print_sample_info(class_name, attr, True, attr_sample_size, attr_available)
                    attr_sample_size = min(attr_sample_size, attr_available)
                    
                    attr_df = class_df[class_df[attr] == 1].sample(attr_sample_size, replace=False)
                    dfs.append(attr_df)

        self.attr_df = pd.concat(dfs)
        print("\n")

    def print_sample_info(self, class_name, attr, is_true, sample_size, available):
        """
        Prints information about the sampled data.

        Args:
            class_name (str): The name of the class.
            attr (str): The attribute.
            is_true (bool): Whether the attribute is true or false.
            sample_size (int): The desired sample size.
            available (int): The number of available samples.
        """
        attr_state = "" if is_true else "Not"
        if sample_size > available:
            print(f"DataLoader Warning: Class-'{class_name}' Att-'{attr_state} {attr}' - desired sample size is {sample_size}, but only {available} samples are available.")
        else:
            print(f"Class-'{class_name}' Att-'{attr_state} {attr}': Sampled {sample_size} images.")

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.attr_df.iloc[idx]['image_id'])
        image = Image.open(img_name)

        # Apply mask if option is set
        if self.mask:
            for region in self.mask_region:
                x = self.landmark_df.iloc[idx][region+'_x']
                y = self.landmark_df.iloc[idx][region+'_y']
                
                # Here, you can adjust the mask size (e.g., 10 pixels)
                image = apply_mask(image, x, y, size=10)

        if self.transform:
            image_transformed = self.transform(image)

        label = self.attr_df.iloc[idx][self.class_attr]
        attributes = self.attr_df.iloc[idx][self.imbalance_attr[0]]

        if self.partition == 2:
            return image_transformed, label, attributes, img_name
        
        if self.transform1:
            x1 = self.transform1(image)
            return image_transformed, label, x1
        
        return image_transformed, label

def apply_mask(image, x, y, size):
    """Applies a mask to a region of an image."""
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x - size//2, y - size//2), (x + size//2, y + size//2)], fill="black")
    return image

def plot_attribute_distribution(dataset, selected_attribute):
    fig, axes = plt.subplots(2, 1, figsize=(15,10))

    # Filter the dataframe to include only images that have the selected attribute
    selected_df = dataset.attr_df[dataset.attr_df[selected_attribute] == 1]
    not_selected_df = dataset.attr_df[dataset.attr_df[selected_attribute] == 0]

    # Drop 'image_id' column from the dataframe
    selected_df = selected_df.drop('image_id', axis=1)
    not_selected_df = not_selected_df.drop('image_id', axis=1)

    # Get the sum of all other attributes
    attribute_sums_selected = selected_df.drop(selected_attribute, axis=1).sum()
    attribute_sums_not_selected = not_selected_df.drop(selected_attribute, axis=1).sum()

    # Create color palette
    color_palette = ['b' if attr != dataset.imbalance_attr else 'r' for attr in attribute_sums_selected.index]

    # Create bar plots
    sns.barplot(x=attribute_sums_selected.index, y=attribute_sums_selected.values, ax=axes[0], palette=color_palette)
    sns.barplot(x=attribute_sums_not_selected.index, y=attribute_sums_not_selected.values, ax=axes[1], palette=color_palette)

    # Set plot properties
    total_selected = selected_df.shape[0]
    total_not_selected = not_selected_df.shape[0]
    axes[0].set_title(f"Distribution of attributes for '{selected_attribute}' (Total count: {total_selected})")
    axes[1].set_title(f"Distribution of attributes for 'Not {selected_attribute}' (Total count: {total_not_selected})")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
    axes[0].set_ylabel('Count')
    axes[1].set_ylabel('Count')

    # Add horizontal grid
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('att_dist.jpg')

def get_celebA_dataloader(batch_size, class_attr, imbalance_attr, imbalance_percent, ignore_attrs, img_size=224, mask=None, mask_region=None):
    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # class_attr = 'Young' # attribute for binary classification
    # imbalance_attr = ['Male']
    # imbalance_percent = {1: [80], 0:[20]} # 1 = Young, 0 = Not Young; 80% of the Young data will be Male
    # ignore_attrs = []  # Example: ignore samples that are 'Bald' or 'Wearing_Earrings'
    # mask = True
    # # mask_region = ['lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth'] 
    # mask_region = []

    # Load the dataset
    train_dataset = FilteredCelebADataset(root_dir='data/CelebA/CelebA/Img/img_align_celeba/', 
                                           transform=data_transforms, 
                                           split='tr', 
                                           class_attr=class_attr, 
                                           num_images=62030,
                                           imbalance_attr=imbalance_attr,
                                           imbalance_percent=imbalance_percent,
                                           ignore_attrs=ignore_attrs,
                                           mask=mask,
                                           mask_region=mask_region)
    
    val_dataset = FilteredCelebADataset(root_dir='data/CelebA/CelebA/Img/img_align_celeba/',
                                        transform=data_transforms,
                                        split='va',
                                        num_images=7180,
                                        class_attr=class_attr,
                                        imbalance_attr=imbalance_attr,
                                        imbalance_percent={1: [50], 0:[50]} ,
                                        ignore_attrs=ignore_attrs)

    test_dataset = FilteredCelebADataset(root_dir='data/CelebA/CelebA/Img/img_align_celeba/',
                                        transform=data_transforms,
                                        split='te',
                                        num_images=31015,
                                        class_attr=class_attr,
                                        imbalance_attr=imbalance_attr,
                                        imbalance_percent={1: [50], 0:[50]} ,
                                        ignore_attrs=ignore_attrs)

    # Define a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    class_names = ['Not Young', 'Young']
    return loaders, class_names

if __name__ == '__main__':

    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    class_attr = 'Young' # attribute for binary classification
    imbalance_attr = ['Male']
    imbalance_percent = {1: [20], 0:[80]} # 1 = Young, 0 = Not Young; 20% of the Young data will be Male
    ignore_attrs = []  # Example: ignore samples that are 'Bald' or 'Wearing_Earrings'
    # mask = True
    # mask_region = ['lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth'] 
    mask = False
    mask_region = []

    loaders, class_names = get_celebA_dataloader(batch_size=4, class_attr=class_attr, imbalance_attr=imbalance_attr, 
                                                imbalance_percent=imbalance_percent, ignore_attrs=ignore_attrs, 
                                                mask=mask, mask_region=mask_region)
    

    # Get a batch of images and labels
    images, labels, _ = next(iter(loaders['train']))

    # Save image
    # Unnormalize the image
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    save_image(torchvision.utils.make_grid(images), 'batch_image.png')
    
    
    # Print the class names for the labels
    for label in labels:
        print(class_names[label])


    # Plot the attribute distribution for 'Young'
    # plot_attribute_distribution(celeba_dataset, class_attr)

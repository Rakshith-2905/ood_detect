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
    def __init__(self, root, split, transform1=None, transform2=None, data_type='imagenet', **kwargs):
        super().__init__(root, **kwargs)
        self.transform1 = transform1
        self.transform2 = transform2

        # Get the class names from the IN_CLASS_MAPPING
        class_name_to_idx = {v[-1]: k for k, v in IN_CLASS_MAPPING.items()}
        wnids_to_idx = {v[0]: k for k, v in IN_CLASS_MAPPING.items()}
        wnids_to_class_names = {v[0]: v[-1] for v in IN_CLASS_MAPPING.values()}
        class_names_to_wnids = {v[-1]: v[0] for v in IN_CLASS_MAPPING.values()}

        self.class_names = list(wnids_to_class_names.values())

        if data_type == 'imagenet' or data_type == 'imagenet-val' or data_type == 'imagenetv2':
            self.wnids = all_wnids
        elif data_type == 'imagenet_r':
            self.wnids = imagenet_r_wnids
            self.class_names = [wnids_to_class_names[wnid] for wnid in self.wnids]            
        elif data_type == 'imagenet_a':
            self.wnids = imagenet_a_wnids
            # self.class_names = [wnids_to_class_names[wnid] for wnid in self.wnids]
        elif data_type == 'imagenet_sketch':
            pass
        else:
            raise ValueError(f'Unknown data_type: {data_type}')
        self.split = split


    def __getitem__(self, index):
        image, label = super(ImageNetTwoTransforms, self).__getitem__(index)

        primary_image = self.transform1(image) if self.transform1 else image
        secondary_image = self.transform2(image) if self.transform2 else image
        if self.transform2 is None:
            return primary_image, label
        return primary_image, label, secondary_image

class ImageNetTwoTransforms_subclasses(ImageFolder):
    def __init__(self, root, split, transform1=None, transform2=None, data_type='imagenet', num_classes=None, **kwargs):
        super().__init__(root, **kwargs)
        self.transform1 = transform1
        self.transform2 = transform2

        # Get the class names from the IN_CLASS_MAPPING
        class_name_to_idx = {v[-1]: k for k, v in IN_CLASS_MAPPING.items()}
        self.idx_to_class_name = {k: v[-1] for k, v in IN_CLASS_MAPPING.items()}
        wnids_to_idx = {v[0]: k for k, v in IN_CLASS_MAPPING.items()}
        self.wnids_to_class_names = {v[0]: v[-1] for v in IN_CLASS_MAPPING.values()}
        class_names_to_wnids = {v[-1]: v[0] for v in IN_CLASS_MAPPING.values()}

        self.class_names = list(self.wnids_to_class_names.values())
        self.class_to_idx = class_name_to_idx

        self.split = split

        if num_classes:
            self._select_random_classes(num_classes)

        if data_type == 'imagenet' or data_type == 'imagenet-val' or data_type == 'imagenetv2':
            self.wnids = all_wnids
                
        elif data_type == 'imagenet_r':
            self.wnids = imagenet_r_wnids
            self.class_names = [self.wnids_to_class_names[wnid] for wnid in self.wnids]

        elif data_type == 'imagenet_a':
            self.wnids = imagenet_a_wnids
            # self.class_names = [wnids_to_class_names[wnid] for wnid in self.wnids]
        elif data_type == 'imagenet_sketch':
            pass
        else:
            raise ValueError(f'Unknown data_type: {data_type}')
        
    def _select_random_classes(self, num_classes):
        
        # Select the classes that are used in imagenet_r
        selected_classes = [self.wnids_to_class_names[wnid] for wnid in imagenet_r_wnids]
        
        # Get the indices for the selected classes
        selected_indices = [int(self.class_to_idx[cls]) for cls in selected_classes]

        selected_indices = sorted(selected_indices)
        selected_classes = [self.idx_to_class_name[str(idx)] for idx in selected_indices]        

        print(f"Selected classes: {selected_indices}")

        # Filter the samples to only include the selected classes
        self.samples = [sample for sample in self.samples if sample[1] in selected_indices]
        self.targets = [target for target in self.targets if target in selected_indices]

        
        # Update classes and class_to_idx to reflect only the selected classes
        self.classes = selected_classes
        self.class_to_idx_new = {cls: idx for idx, cls in enumerate(selected_classes)}

        self.class_names = selected_classes

        # Map dictionary from new class indices to original class indices
        self.old_idx_to_new_idx = {int(self.class_to_idx[cls]): new_idx for cls, new_idx in self.class_to_idx_new.items()}

        
    def __getitem__(self, index):
        image, label = super(ImageNetTwoTransforms_subclasses, self).__getitem__(index)

        label = self.old_idx_to_new_idx[label]
        primary_image = self.transform1(image) if self.transform1 else image
        secondary_image = self.transform2(image) if self.transform2 else image
        if self.transform2 is None:
            return primary_image, label
        return primary_image, label, secondary_image

def plot_images(loader, title, n_rows=2, n_cols=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], class_names=None):
    """
    Extracts the first batch of images from the given data loader and plots them in a grid with their labels.
    Adjusts the image contrast if necessary.
    """
    import matplotlib.pyplot as plt

    # Get the first batch
    images, labels = next(iter(loader))


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axes = axes.flatten()

    # Function to denormalize the image
    def denormalize(image):
        image = image.numpy().transpose(1, 2, 0)
        image = std * image + mean
        image = np.clip(image, 0, 1)
        return image

    for i in range(n_rows * n_cols):
        image = denormalize(images[i])
        label = labels[i]
        print("label",label)
        class_name = class_names[label]


        axes[i].imshow(image)
        axes[i].set_title(f"Label: {class_name}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(f"{title}.png")
    plt.savefig(save_path)
    # plt.show()

# def get_imagenet_loaders(batch_size=512, data_dir='./data',    
#                         train_transform=None, test_transform=None, clip_transform=None, 
#                         data_type='imagenet', subsample_trainset=True, return_dataset=False):
    
#     if train_transform is None:
#         train_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#     if test_transform is None:
#         test_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])


#     if train_transform is None:
#         train_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#     if test_transform is None:
#         test_transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])


#     if data_type == 'imagenet':
#         if os.path.exists(os.path.join(data_dir, 'imagenet', 'train')):
#             train_data_dir = os.path.join(data_dir, 'imagenet', 'train')
#             test_data_dir = os.path.join(data_dir, 'imagenet', 'val')
#         else:
#             train_data_dir = os.path.join(data_dir,  'train')
#             test_data_dir = os.path.join(data_dir,  'val')

#         train_dataset = ImageNetTwoTransforms_subclasses(root=train_data_dir, split='train',transform1=train_transform, 
#                                                 transform2=clip_transform, data_type=data_type, num_classes=200)
        
#         print("train_dataset",len(train_dataset))
        
#         class_names = train_dataset.class_names
#         # Split the test set into val and test
#         temp_testset = ImageNetTwoTransforms_subclasses(root=test_data_dir, split='val',transform1=test_transform,
#                                                 transform2=clip_transform, data_type=data_type, num_classes=200)
#         print("temp_testset",len(temp_testset))
#         val_size = int(0.10 * len(temp_testset))
#         test_size = len(temp_testset) - val_size
#         test_dataset, val_dataset = torch.utils.data.random_split(temp_testset, [test_size, val_size], 
#                                                                     generator=torch.Generator().manual_seed(42))
        
#         print("test_dataset",len(test_dataset))
#         # Split the val set into val and failure
#         failure_size = int(0.25 * len(val_dataset))
#         val_size = len(val_dataset) - failure_size
#         val_dataset, failure_dataset = torch.utils.data.random_split(val_dataset, [val_size, failure_size], 
#                                                                     generator=torch.Generator().manual_seed(42))
        
#         print("val_dataset",len(val_dataset))
        
#     else:

#         val_data_dir = os.path.join(data_dir, 'imagenet', 'val')
#         test_data_dir = os.path.join(data_dir, 'imagenet', data_type)

#         val_dataset = ImageNetTwoTransforms_subclasses(root=val_data_dir, split='val',transform1=train_transform,
#                                                 transform2=clip_transform, data_type=data_type, num_classes=200)
        
#         test_dataset = ImageNetTwoTransforms_subclasses(root=test_data_dir, split='test',transform1=test_transform,
#                                                 transform2=clip_transform, data_type=data_type, num_classes=200)
        
#         class_names = val_dataset.class_names
        
#         # # Split the val set into val and failure
#         # val_size = int(0.10 * len(val_dataset))
#         # unwanted_size = len(val_dataset) - val_size
#         # val_dataset, unwanted_dataset = torch.utils.data.random_split(val_dataset, [val_size, unwanted_size], 
#         #                                                             generator=torch.Generator().manual_seed(42))

#         # # split the val set into val and failure
#         # failure_size = int(0.25 * len(val_dataset))
#         # val_size = len(val_dataset) - failure_size
#         # val_dataset, failure_dataset = torch.utils.data.random_split(val_dataset, [val_size, failure_size],
#         #                                                             generator=torch.Generator().manual_seed(42))


#         # This is just to make the code work
#         train_dataset = val_dataset
#         failure_dataset = val_dataset
#         # train_data_dir = os.path.join(data_dir, 'imagenet', 'train')
#         # train_dataset = ImageNetTwoTransforms_subclasses(root=train_data_dir, split='train',transform1=train_transform, 
#         #                                 transform2=clip_transform, data_type=data_type, num_classes=200)

#     if return_dataset:
#         return train_dataset, val_dataset, test_dataset, failure_dataset, class_names

#     # DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     failure_loader = DataLoader(failure_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#     loaders = {
#         'train': train_loader,
#         'val': val_loader,
#         'failure': failure_loader,
#         'test': test_loader
#     }
#     return loaders, class_names


def get_imagenet_loaders(batch_size=512, data_dir='./data',    
                        train_transform=None, test_transform=None, clip_transform=None, 
                        data_type='imagenet', subsample_trainset=True, return_dataset=False):
    
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    if data_type == 'imagenet':
        if os.path.exists(os.path.join(data_dir, 'imagenet', 'train')):

            train_data_dir = os.path.join(data_dir, 'imagenet', 'train')
            test_data_dir = os.path.join(data_dir, 'imagenet', 'val')
        else:
            train_data_dir = os.path.join(data_dir,  'train')
            test_data_dir = os.path.join(data_dir,  'val')
    else:
        train_data_dir = os.path.join(data_dir, 'imagenet', 'val')
        test_data_dir = os.path.join(data_dir, 'imagenet', data_type)

    train_dataset = ImageNetTwoTransforms(root=train_data_dir, split='train',transform1=train_transform, 
                                               transform2=clip_transform, data_type=data_type)
    
    val_dataset = train_dataset
    failure_dataset = train_dataset

    # temp_valset = train_dataset

    # # Split the valset into val and failure
    # failure_size = int(0.50 * len(temp_valset))
    # val_size = len(temp_valset) - failure_size
    # val_dataset, failure_dataset = torch.utils.data.random_split(temp_valset, [val_size, failure_size], 
    #                                                             generator=torch.Generator().manual_seed(42))
    
    if data_type == 'imagenet':
        test_dataset = ImageNetTwoTransforms(root=test_data_dir, split='val',transform1=test_transform, 
                                            transform2=clip_transform, data_type=data_type)
    elif data_type == 'imagenet-val':
        test_dataset = val_dataset
    else:
        test_dataset = ImageNetTwoTransforms(root=test_data_dir, split='test',transform1=test_transform, 
                                            transform2=clip_transform, data_type=data_type)

    if return_dataset:
        return train_dataset, val_dataset, test_dataset, failure_dataset, train_dataset.class_names

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
    return loaders, train_dataset.class_names

if __name__ == "__main__":

    loaders, class_names = get_imagenet_loaders(batch_size=512, data_dir='../data',
                                                train_transform=None, test_transform=None, clip_transform=None,
                                                data_type='imagenet_sketch', subsample_trainset=False, return_dataset=False)
    
    print("Statistics")
    print("Train:", len(loaders['train'].dataset))
    print("Val:", len(loaders['val'].dataset))
    print("Failure:", len(loaders['failure'].dataset))
    print("Test: ", len(loaders['test'].dataset))
    print(len(class_names))
    print(class_names)

    # get the first batch of images
    plot_images(loaders['train'], 'imagenet_Train Set', class_names=class_names)

    # plot_images(loaders['train'], 'imagenet_a_Train Set')
    # plot_images(loaders['val'], 'imagenet_a_Val Set')
    # plot_images(loaders['test'], 'imagenet_a_Test Set')
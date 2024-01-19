import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms


class DomainNetDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', transform=None, transform2=None):
        self.root_dir = root_dir
        self.domain = domain # domain name
        self.split = split   # 'train' or 'test'
        self.transform = transform
        self.transform2 = transform2
        
        # Load file list
        file_list = os.path.join(root_dir, f'text_files/{domain}_{split}.txt') 
        with open(file_list) as f:
            self.file_paths = [line.strip() for line in f] 
        self.targets = [int(path.split()[1]) for path in self.file_paths]
        # only keep unique labels
        self.targets =np.array(list(set(self.targets)))
        self.class_names = [line.strip() for line in open(os.path.join(root_dir, 'class_names.txt'))]

        self.num_classes = len(self.targets)
        print(f"Number of classes in {domain} {split} set: {self.num_classes}")
        print(f"Number of images in {domain} {split} set: {len(self.file_paths)}\n")
    def __len__(self):
        return len(self.file_paths)
            
    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx].split() 
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        image2 = image.copy()
        # File contains image path and label
        label = int(label) 

        if self.transform:
            image = self.transform(image)
        if self.transform2:
            image2 = self.transform2(image2)
            return image, label, image2
        
        return image, label

    def undo_transformation(self, images):
        # Undo the transformation on the inputs
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        return inv_normalize(images)

def get_data_from_saved_files(save_dir, batch_size=32, train_shuffle=True, return_dataset=False,dataset_name='domainnet'):

    train_outputs = torch.load(os.path.join(save_dir, "train_outputs.pth"))
    train_CLIP_features = torch.load(os.path.join(save_dir, "train_ViTB32_CLIP_features.pth"))
    train_features = torch.load(os.path.join(save_dir, "train_features.pth"))
    train_labels = torch.load(os.path.join(save_dir, "train_labels.pth"))

    test_outputs = torch.load(os.path.join(save_dir, "test_outputs.pth"))
    test_CLIP_features = torch.load(os.path.join(save_dir, "test_ViTB32_CLIP_features.pth"))
    test_features = torch.load(os.path.join(save_dir, "test_features.pth"))
    test_labels = torch.load(os.path.join(save_dir, "test_labels.pth"))

    # Create TensorDatasets
    train_dataset = TensorDataset(train_outputs, train_features, train_labels, train_CLIP_features)
    test_dataset = TensorDataset(test_outputs, test_features, test_labels, test_CLIP_features)

    # load class names from data/domainnet_v1.0/class_names.txt
    if dataset_name=='domainnet':
        with open('./data/domainnet_v1.0/class_names.txt') as f:
            class_names = [line.strip() for line in f]
    elif 'cifar10' in dataset_name:
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if return_dataset:
        return train_dataset, test_dataset, class_names

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_domainnet_loaders(domain_name, batch_size=512, data_dir='data/', 
                          train_transform=None, test_transform=None, clip_transform=None, 
                          subsample_trainset=False, return_dataset=False):
    
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
    
    data_dir = os.path.join(data_dir, 'domainnet_v1.0')
    temp_train_dataset = DomainNetDataset(root_dir=data_dir, domain=domain_name,
                                    split='train', transform=train_transform, transform2=clip_transform)
    test_dataset= DomainNetDataset(root_dir=data_dir, domain=domain_name, 
                                    split='test', transform=test_transform, transform2=clip_transform)
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

    
if __name__=="__main__":

    train_transform = None
    test_transform = None
    clip_transform = None
    datasets = get_domainnet_loaders('real', batch_size=4, data_dir='data', 
                            train_transform=train_transform, test_transform=test_transform, clip_transform=clip_transform,
                            subsample_trainset=False, return_dataset=True)

    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = datasets

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    print(len(failure_dataset))
    print(class_names)
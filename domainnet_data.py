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
    elif dataset_name=='cifar10_full':
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if return_dataset:
        return train_dataset, test_dataset, class_names

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_domainnet_loaders(domain_name,batch_size=512,train_shuffle=True, data_dir='data/domainnet_v1.0'):
    imagenet_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    imagenet_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])
    domain_train = DomainNetDataset(root_dir=data_dir, domain=domain_name, \
                                    split='train', transform=imagenet_train_transform)
    domain_test= DomainNetDataset(root_dir=data_dir, domain=domain_name, \
                                    split='test', transform=imagenet_test_transform)
    train_loader = torch.utils.data.DataLoader(domain_train, batch_size=batch_size, shuffle=train_shuffle, num_workers=8)
    test_loader = torch.utils.data.DataLoader(domain_test, batch_size=batch_size, shuffle=False, num_workers=8)
    loaders={}
    loaders['train']=train_loader
    loaders['test']=test_loader
    class_names = domain_train.targets
    return loaders, class_names

    
if __name__=="__main__":
    transform_train= transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ])


    real_train = DomainNetDataset(root_dir='data/domainnet_v1.0', domain='real', split='train', transform=transform_train)
    print(len(real_train))
    print(real_train.class_names)
    assert False
    loader = torch.utils.data.DataLoader(real_train, batch_size=4, shuffle=True, num_workers=2)
    for i, data in enumerate(loader):
        print(data[0].shape)
        print(data[1].shape)
        if i==2:
            break
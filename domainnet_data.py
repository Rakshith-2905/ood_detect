import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms

class DomainNetDataset(Dataset):
    def __init__(self, root_dir, domain, split='train', transform=None):
        self.root_dir = root_dir
        self.domain = domain # domain name
        self.split = split   # 'train' or 'test'
        self.transform = transform
        
        # Load file list
        file_list = os.path.join(root_dir, f'text_files/{domain}_{split}.txt') 
        with open(file_list) as f:
            self.file_paths = [line.strip() for line in f] 
        self.targets = [int(path.split()[1]) for path in self.file_paths]
        # only keep unique labels
        self.targets =np.array(list(set(self.targets)))

        self.num_classes = len(self.targets)
        print(f"Number of classes in {domain} {split} set: {self.num_classes}")
        print(f"Number of images in {domain} {split} set: {len(self.file_paths)}\n")
    def __len__(self):
        return len(self.file_paths)
            
    def __getitem__(self, idx):
        img_path, label = self.file_paths[idx].split() 
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        # File contains image path and label
        label = int(label) 
        
        return image, label

def get_domainnet_loaders(domain_name,batch_size=512,train_shuffle=True):
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
    domain_train = DomainNetDataset(root_dir='data/domainnet_v1.0', domain=domain_name, \
                                    split='train', transform=imagenet_train_transform)
    domain_test= DomainNetDataset(root_dir='data/domainnet_v1.0', domain=domain_name, \
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
    loader = torch.utils.data.DataLoader(real_train, batch_size=4, shuffle=True, num_workers=2)
    for i, data in enumerate(loader):
        print(data[0].shape)
        print(data[1].shape)
        if i==2:
            break
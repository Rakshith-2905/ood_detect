import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class PACSDataset(Dataset):
    def __init__(self, data_dir, domain, split='train', transform1=None, transform2=None):
        """
        Args:
            txt_file (string): Path to the text file with annotations.
            data_dir (string): Directory with all the images.
            domain (string): The domain name associated with the dataset.
            transform1 (callable, optional): Optional transform to be applied on a sample.
            transform2 (callable, optional): Second optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        
        annotations_file = os.path.join(data_dir, 'pacs_label', f"{domain}_{split}.txt")
        self.domain = domain

        self.transform1 = transform1
        self.transform2 = transform2

        self.class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

        self.labels = []
        self.img_names = []
        with open(annotations_file, 'r') as file:
            for line in file:
                line = line.strip().split()
                self.img_names.append(line[0])
                self.labels.append(int(line[1]))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_dir, 'pacs_data', self.img_names[idx])  # Adjusted to include domain in path
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB
        label = self.labels[idx] - 1  # Adjust label to start from 0

        if self.transform1:
            image = self.transform1(image)

        if self.transform2:
            return image, label, self.transform2(image)
        
        return image, label

def get_pacs_dataloader(domain_name, batch_size=512, data_dir='data/', 
                        train_transform=None, test_transform=None, clip_transform=None, 
                        return_dataset=False, use_real=True):
    
    available_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    assert domain_name in available_domains, f"Domain name must be one of {available_domains}"
    
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
    
    train_domain_name = domain_name
    if use_real:
        train_domain_name = 'photo'
    
    data_dir = os.path.join(data_dir, 'pacs')
    train_dataset = PACSDataset(data_dir=data_dir, domain=train_domain_name,
                                    split='train', transform1=train_transform, transform2=clip_transform)
    test_dataset= PACSDataset(data_dir=data_dir, domain=domain_name,
                                    split='test', transform1=test_transform, transform2=clip_transform)
    temp_valset = PACSDataset(data_dir=data_dir, domain=train_domain_name,
                                    split='val', transform1=train_transform, transform2=clip_transform)

    # Split the valset into val and failure
    failure_size = int(0.50 * len(temp_valset))
    val_size = len(temp_valset) - failure_size
    val_dataset, failure_dataset = torch.utils.data.random_split(temp_valset, [val_size, failure_size], 
                                                                 generator=torch.Generator().manual_seed(42))

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
    # Test the dataloader
    loaders, class_names = get_pacs_dataloader('photo', return_dataset=False)
    print(class_names)
    print(len(loaders['train'].dataset))
    print(len(loaders['val'].dataset))
    print(len(loaders['test'].dataset))
    print(len(loaders['failure'].dataset))

    for i, (images, labels) in enumerate(loaders['train']):
        print(images.shape)
        print(labels)
        break   
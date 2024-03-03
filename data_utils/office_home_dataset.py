import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import pandas as pd

class OfficeHomeDataset(Dataset):
    def __init__(self, data_dir, domain, split, transform1=None, transform2=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            domain (string): One of 'Art', 'Clipart', 'Product', 'Real_World'.
            split (string): One of 'train', 'val', 'test'.
            transform1 (callable, optional): Optional transform to be applied on a sample.
            transform2 (callable, optional): An additional optional transform to be applied on a sample.
        """
        self.data_dir = os.path.join(data_dir, domain)
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_paths = []
        self.labels = []
        
        # Load file list
        file_list = os.path.join(data_dir, f'{domain}.txt')
        
        df = pd.read_csv(file_list)
        # Slect the rows based on the split
        df = df[df['split'] == split]
        self.image_paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()

        # Load the class names
        class_names_file = os.path.join(data_dir, f'class_names.txt')
        with open(class_names_file, 'r') as f:
            self.class_names = [line.strip() for line in f]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform1:
            image = self.transform1(image)

        if self.transform2:
            image1 = self.transform2(image)
            return image, label, image1
        else:
            return image, label


def get_office_home_dataloader(domain_name, batch_size=512, data_dir='data/', 
                        train_transform=None, test_transform=None, clip_transform=None, 
                        return_dataset=False, use_real=True):
    
    available_domains = ['Art', 'Clipart', 'Product', 'Real_World']
    assert domain_name in available_domains, f"Domain name must be one of {available_domains}"
    
    if train_transform == None:
        train_transform=  transforms.Compose([
            transforms.Resize((224, 224)),               
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        
    if test_transform == None:
        test_transform=  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    data_dir = os.path.join(data_dir, 'OfficeHomeDataset_10072016')

    train_domain_name = domain_name
    if use_real:
        train_domain_name = 'Real_World'
    
    train_dataset = OfficeHomeDataset(data_dir=data_dir, domain=train_domain_name,
                                    split='train', transform1=train_transform, transform2=clip_transform)
    test_dataset= OfficeHomeDataset(data_dir=data_dir, domain=domain_name,
                                    split='test', transform1=test_transform, transform2=clip_transform)
    temp_valset = OfficeHomeDataset(data_dir=data_dir, domain=train_domain_name,
                                    split='val', transform1=test_transform, transform2=clip_transform)


    # Split the valset into val and failure
    failure_size = int(0.50 * len(temp_valset))
    val_size = len(temp_valset) - failure_size
    val_dataset, failure_dataset = torch.utils.data.random_split(temp_valset, [val_size, failure_size], 
                                                                 generator=torch.Generator().manual_seed(42))

    if return_dataset:
        return train_dataset, val_dataset, test_dataset, None, train_dataset.class_names

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

def split_data(root_dir):

    domain_names = ['Art', 'Clipart', 'Product', 'Real_World']

    # For every domain split the data into train, val, test and save to text files
    for domain_name in domain_names:
        data_dir = os.path.join(root_dir, domain_name)

        image_paths = []
        labels = []
        class_names = sorted(os.listdir(data_dir))
        label_map = {class_name: idx for idx, class_name in enumerate(class_names)}

        # Save the class names to a file
        with open(os.path.join(root_dir, f'class_names.txt'), 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        # Iterate through each class directory and collect image paths and labels
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(label_map[class_name])

        # Randomly shuffle the data together
        combined = list(zip(image_paths, labels))
        # seed random number generator
        np.random.seed(42)
        np.random.shuffle(combined)
        image_paths[:], labels[:] = zip(*combined)


        # Split the data into train, val, test using sklearn
        from sklearn.model_selection import train_test_split
        train_image_paths, val_test_image_paths, train_labels, val_test_labels = train_test_split(image_paths, labels, test_size=0.20, random_state=42)

        val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(val_test_image_paths, val_test_labels, test_size=0.50, random_state=42)


        print(f"{domain_name} Train: {len(train_image_paths)}, Val: {len(val_image_paths)}, Test: {len(test_image_paths)}")

        # Save the data to csv files as image_path, label, split
        with open(os.path.join(root_dir, f'{domain_name}.txt'), 'w') as f:
            f.write("image_path,label,split\n")
            for img_path, label in zip(train_image_paths, train_labels):
                f.write(f"{img_path},{label},train\n")

            for img_path, label in zip(val_image_paths, val_labels):
                f.write(f"{img_path},{label},val\n")

            for img_path, label in zip(test_image_paths, test_labels):
                f.write(f"{img_path},{label},test\n")

if __name__ == "__main__":

    # Split the data into train, val, test and save to text files
    split_data('data/OfficeHomeDataset_10072016')

    # Test the dataloader
    loaders, class_names = get_office_home_dataloader('Art', return_dataset=False, use_real=False)
    print(class_names)
    print(len(loaders['train'].dataset))
    print(len(loaders['val'].dataset))
    print(len(loaders['test'].dataset))
    print(len(loaders['failure'].dataset))

    for i, (images, labels) in enumerate(loaders['train']):
        print(images.shape)
        print(labels)
# 
        break






        

    

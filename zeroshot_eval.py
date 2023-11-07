import torch
import clip
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
import json
import argparse
from tqdm import tqdm
import os

from cifar100_data import get_CIFAR100_loaders
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow, NumpyDataLoader
from models.resnet import CustomResNet
from models.visual_transformer import ProjectionHead


class CIFAR100CDataset(Dataset):
    def __init__(self, np_images, np_labels, transform=None):
        self.images = np_images
        self.labels = np_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Convert to PIL Image to apply torchvision transforms
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

def get_dataloader(dataset_name, preprocess, train=False):
    if dataset_name.lower() == 'cifar100':
        dataset = CIFAR100(root="./data", download=True, transform=preprocess, train=train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    # Add imagenet here
    elif dataset_name.lower() == 'imagenet':
        dataset = ImageNet(root="./data", download=True, transform=preprocess, train=train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    elif dataset_name.lower() == 'cifar100-c':
        # Load the numpy files
        cifar100_c_path = "data/CIFAR100-C"
        
        dataloader = {}
        np_labels = np.load(os.path.join(cifar100_c_path, "labels.npy"))
        # tensor_labels = torch.Tensor(np_labels)

        # Sort the files in the directory
        corruptions_path = os.listdir(cifar100_c_path)
        corruptions_path.sort()

        for file in corruptions_path:
            if 'labels' in file:
                continue
            if file.endswith(".npy"):
                np_images = np.load(os.path.join(cifar100_c_path, file))
                # data = zip(np_images, np_labels)

                # dataloader[file.split(".")[0]] = data
                # tensor_images = torch.Tensor(np_images)
                # dataset = TensorDataset(tensor_images, tensor_labels)
                # dataloader[file.split(".")[0]] = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                # del np_images

                dataset = CIFAR100CDataset(np_images, np_labels, transform=preprocess)
                dataloader[file.split(".")[0]] = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    elif dataset_name.lower() == 'cifar100-10':
        # Sorting the classes
        np.random.seed(42)
        select_indices = sorted(np.random.choice(list(range(100)), 90, replace=False))
        remaining_indices = list(set(list(range(100))) ^ set(select_indices))
        remaining_indices.sort()

        # Get the loaders and class names
        loaders, class_names = get_CIFAR100_loaders(batch_size=args.batch_size, data_dir='./data', \
                                select_indices=remaining_indices, retain_orig_ids=False, 
                                train_transform=preprocess, test_transform=preprocess)
        dataloader = loaders['test']

        return dataloader, remaining_indices, class_names
    
    elif dataset_name.lower() == 'cifar100-90':
        # Sorting the classes
        np.random.seed(42)
        select_indices = sorted(np.random.choice(list(range(100)), 90, replace=False))
        remaining_indices = list(set(list(range(100))) ^ set(select_indices))
        remaining_indices.sort()

        # Get the loaders and class names
        loaders, class_names = get_CIFAR100_loaders(batch_size=args.batch_size, data_dir='./data', \
                                select_indices=select_indices, retain_orig_ids=False, 
                                train_transform=preprocess, test_transform=preprocess)
        dataloader = loaders['test']

        return dataloader, select_indices, class_names
    

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return dataloader

def evaluate_CLIP(model, dataloader, text_encodings, device):
    model.eval()
    total_accuracy = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            # Compute image embeddings
            image_features = model.encode_image(images)

            # Compute similarities between image embeddings and text encodings
            similarities = compute_similarities(image_features, text_encodings, mode="cosine")

            # Compute the predictions
            probs = F.softmax(similarities, dim=1)
            predictions = torch.argmax(probs, dim=1)

            # Compute the accuracy
            total_accuracy += compute_accuracy(probs, labels)

            # Save the predictions and labels
            all_preds.append(predictions)
            all_labels.append(labels)
            all_probs.append(probs)

    return total_accuracy/len(all_preds), all_preds, all_labels, all_probs

def evaluate_projection(resnet_model, projector, dataloader, text_encodings, device):
    resnet_model.eval()
    projector.eval()
    total_accuracy = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            # Compute the ResNet embeddings
            resnet_logits, resnet_embeddings = resnet_model(images, return_features=True)
            probs_from_resnet = F.softmax(resnet_logits, dim=-1)
            
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)

            # Compute similarities between the projected embeddings and text encodings
            similarities = compute_similarities(proj_embeddings, text_encodings, mode="cosine")

            # Compute the predictions
            probs = F.softmax(similarities, dim=1)
            predictions = torch.argmax(probs, dim=1)

            # Compute the accuracy
            total_accuracy += compute_accuracy(probs, labels)

            # Save the predictions and labels
            all_preds.append(predictions)
            all_labels.append(labels)
            all_probs.append(probs)

    return total_accuracy/len(all_preds), all_preds, all_labels, all_probs

def main(args):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_model, preprocess = clip.load("RN50", device=device)

    if args.model_type == 'clip':
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                         std=[0.26862954, 0.26130258, 0.27577711]),
        # ])
        # CLIP_preproces = transforms.Compose([
        #     transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #                             std=[0.26862954, 0.26130258, 0.27577711]),
        # ])
        transform = preprocess
    elif args.model_type == 'projection':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Load the dataset
    dataloader = get_dataloader(args.dataset_name, transform)
    text_encodings = torch.load(args.text_encoding_path)[0]

    # if dataloader is a tuple, unpack it
    if isinstance(dataloader, tuple):
        dataloader, select_indices, class_names = dataloader
        print(f"Selected indices: {select_indices}")

        # Print the number of images in train and test sets
        print(f"Number of train images: {len(dataloader.dataset)}")
        print(f"Number of test images: {len(dataloader.dataset)}")

        # Filter the text encodings
        text_encodings = text_encodings[select_indices]

    if args.model_type == 'projection':
        resnet_model = CustomResNet(model_name='resnet50', num_classes=100, use_pretrained=True).to(device)
        projector = ProjectionHead(input_dim=2048, output_dim=1024).to(device)
        projector.load_state_dict(torch.load(args.projector_checkpoint_path))
        projector.eval()


    with torch.no_grad():
        # if dataloader is a dictionary, iterate over the different items and print the accuracy for each
        if isinstance(dataloader, dict):
            for name, loader in dataloader.items():
                if args.model_type == 'clip':
                    accuracy, all_preds, all_labels, all_probs = evaluate_CLIP(CLIP_model, loader, text_encodings, device)
                elif args.model_type == 'projection':
                    accuracy, all_preds, all_labels, all_probs = evaluate_projection(resnet_model, projector, loader, text_encodings, device)

                print(f"{args.model_type} {args.dataset_name} {name} Accuracy: {accuracy:.6f}")
        else:
            if args.model_type == 'clip':
                accuracy, all_preds, all_labels, all_probs = evaluate_CLIP(CLIP_model, dataloader, text_encodings, device)
            elif args.model_type == 'projection':
                accuracy, all_preds, all_labels, all_probs = evaluate_projection(resnet_model, projector, dataloader, text_encodings, device)
            # accuracy, all_preds, all_labels, all_probs = evaluate(CLIP_model, dataloader, text_encodings, device)

            print(f"{args.model_type} {args.dataset_name} Accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CLIP zero-shot performance on a dataset.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to evaluate on.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader.')
    parser.add_argument('--text_encoding_path', type=str, required=True, 
                        help='Path to the JSON file with text encodings for labels.')
    parser.add_argument('--model_type', type=str, default='clip', choices=['clip', 'projection'])
    parser.add_argument('--projector_checkpoint_path', type=str, help='Path to the projector checkpoint.')

    
    args = parser.parse_args()

    main(args)
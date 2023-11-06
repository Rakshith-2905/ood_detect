import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.resnet import CustomResNet
from domainnet_data import DomainNetDataset, get_domainnet_loaders
from utils import compute_accuracy


def get_all_domainnet_loaders(batch_size=32):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"] 
    loaders_dict = {}
    for domain in domains:
        loaders_dict[domain], class_names = get_domainnet_loaders(domain, batch_size=batch_size)
    return loaders_dict, class_names

def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, features = model(inputs, return_features=True)
            loss = criterion(outputs, labels)

            predictions = F.softmax(outputs, dim=-1)         
            total_loss += loss.item()

            batch_acc = compute_accuracy(predictions, labels)   

            total_acc += batch_acc
            total += labels.size(0)

    accuracy = 100. * total_acc / len(loader)
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

def save_features_and_labels(loader, model, device, save_dir, prefix="train"):
    """
    Saves outputs, features, and labels from the model for a given dataset.

    Args:
    - loader (torch.utils.data.DataLoader): DataLoader for your dataset.
    - model (torch.nn.Module): Your model.
    - device (torch.device): Device to which data should be loaded.
    - save_dir (str): Directory to save the data.
    - prefix (str): Prefix for filenames, e.g., 'train' or 'test'.
    """
    # Initialize CLIP
    clip_model, preprocess = clip.load("RN50", device=device)

    if prefix == "train":
        base_geometry_transform = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip()])
    else:

        base_geometry_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224)])        
    
    resnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

    dataset = DomainNetDataset(root_dir='data/domainnet_v1.0', domain='real', split=prefix, transform=None)

    all_outputs = []
    all_features = []
    CLIP_features = []
    all_labels = []

    with torch.no_grad():
        # for inputs, labels in loader:
        for i in tqdm(range(len(dataset))):
            inputs, labels = dataset[i]

            labels = torch.tensor(labels)
            
            inputs = base_geometry_transform(inputs)

            inputs_resnet = resnet_transform(inputs).unsqueeze(0).to(device)
            inputs_clip = preprocess(inputs).unsqueeze(0).to(device)

            # inputs = inputs.to(device)

            outputs, features = model(inputs_resnet, return_features=True)

            # Get CLIP image features for the inputs
            image_features = clip_model.encode_image(inputs_clip)

            CLIP_features.append(image_features.cpu())

            all_outputs.append(outputs.cpu())
            all_features.append(features.cpu())
            all_labels.append(labels)

    # Concatenate the results
    CLIP_features = torch.cat(CLIP_features, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.stack(all_labels, dim=0)


    print(f"CLIP features shape: {CLIP_features.shape}")
    print(f"Outputs shape: {all_outputs.shape}")
    print(f"Features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")

    save_dir = os.path.join(save_dir, 'features')
    # Save the results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(CLIP_features, os.path.join(save_dir, f"{prefix}_CLIP_features.pth"))
    torch.save(all_outputs, os.path.join(save_dir, f"{prefix}_outputs.pth"))
    torch.save(all_features, os.path.join(save_dir, f"{prefix}_features.pth"))
    torch.save(all_labels, os.path.join(save_dir, f"{prefix}_labels.pth"))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get all the test loaders for each domain
    loaders_dict, class_names = get_all_domainnet_loaders(batch_size=args.batch_size)

    # Load your trained model from checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    
    model = CustomResNet(model_name=args.resnet_model, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    model.to(device)
    model.eval()
    

    # Save results
    save_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    if not os.path.exists(save_dir):
        assert False, f"Directory {save_dir} does not exist"

    save_features_and_labels(loaders_dict['real']['train'], model, device, save_dir, prefix="train")
    save_features_and_labels(loaders_dict['real']['test'], model, device, save_dir, prefix="test")
    assert False
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    for domain, loader in loaders_dict.items():
        loss, acc = evaluate(loader['test'], model, criterion, device)
        results[domain] = {"Loss": loss, "Accuracy": acc}
        print(f"Domain: {domain}\tLoss: {loss:.4f}\tAccuracy: {acc:.2f}%")

    
    avg_ood_acc = 0
    for domain, acc in results.items():
        if domain != 'real':
            avg_ood_acc += acc['Accuracy']
    avg_ood_acc /= 5
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Domain\tLoss\tAccuracy\n")
        for domain, metrics in results.items():
            f.write(f"{domain}\t{metrics['Loss']:.4f}\t{metrics['Accuracy']:.2f}%\n")
        f.write(f"Average OOD\t{avg_ood_acc:.2f}%\n")

def get_accuracy(model, batch_size, device, save_dir):
    loaders_dict = get_all_domainnet_loaders(batch_size=batch_size)
    accuracies = {}
    for domain, loader in loaders_dict.items():
        _, acc = evaluate(loader['test'], model, criterion, device)
        accuracies[domain] = acc
    if save_dir:
        # Calulate average accuracy across domains excluding real
        avg_ood_acc = sum([acc for domain, acc in accuracies.items() if domain != 'real']) / 5

        # Save accuracies
        with open(os.path.join(save_dir, 'accuracies.txt'), 'w') as f:
            f.write("Domain\tAccuracy\n")
            for domain, acc in accuracies.items():
                f.write(f"{domain}\t{acc:.2f}%\n")
            f.write(f"Average OOD\t{avg_acc:.2f}%\n")
    return accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--domain', type=str, required=True, help='Name of the domain to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=1, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--resnet_model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18', help='Type of ResNet model to use')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()
    main(args)


# python test_classifier.py --dataset domainnet --domain real --image_size 224 --batch_size 64 \
#                             --class_percentage 1     --seed 42  
#                             --resnet_model resnet50 --checkpoint_path 'logs/classifier/resnet50_domainnet_real/best_checkpoint.pth'
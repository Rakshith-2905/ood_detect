import torch
import torch.nn as nn

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.resnet import CustomResNet
from domainnet_data import DomainNetDataset, get_domainnet_loaders


def get_all_domainnet_loaders(batch_size=32):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"] 
    loaders_dict = {}
    for domain in domains:
        loaders_dict[domain], class_names = get_domainnet_loaders(domain, batch_size=batch_size)
    return loaders_dict, class_names

def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy

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
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    for domain, loader in loaders_dict.items():
        loss, acc = evaluate(loader['test'], model, criterion, device)
        results[domain] = {"Loss": loss, "Accuracy": acc}
        print(f"Domain: {domain}\tLoss: {loss:.4f}\tAccuracy: {acc:.2f}%")

    # Save results
    save_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    if not os.path.exists(save_dir):
        assert False, f"Directory {save_dir} does not exist"
    
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
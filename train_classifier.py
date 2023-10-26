import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import WildsDataLoader
from models.resnet import CustomResNet
from domainnet_data import DomainNetDataset, get_domainnet_loaders

def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, len(train_loader.dataset)
    
    # Wrap the train_loader with tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Training epoch: {epoch+1}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item() * inputs.size(0)
        total_loss += batch_loss
        _, predicted = outputs.max(1)
        batch_correct = (predicted == labels).sum().item()
        total_correct += batch_correct
        total_samples += inputs.size(0)

        # Set metrics for tqdm
        pbar.set_postfix({"Epoch Loss": total_loss/total_samples, "Epoch Acc": total_correct/total_samples})

    return total_loss/total_samples, total_correct/total_samples


def validate(val_loader, model, criterion, device, epoch):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating epoch: {epoch+1}')
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_loss = loss.item() * inputs.size(0)
            total_loss += batch_loss
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()
            total_correct += batch_correct
            total_samples += inputs.size(0)
            # Set metrics for tqdm
            pbar.set_postfix({"Epoch Loss": total_loss/total_samples, "Epoch Acc": total_correct/total_samples})

    return total_loss/total_samples, total_correct/total_samples

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders, class_names = get_domainnet_loaders(args.domain, batch_size=args.batch_size, train_shuffle=True)

    train_loader = loaders['train']
    val_loader = loaders['test']

    model = CustomResNet(model_name=args.resnet_model, num_classes=len(class_names), use_pretrained=args.use_pretrained)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Make directory for saving results
    save_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save arguments if not resuming
    if not args.resume:
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        
        start_epoch = 0
        
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        best_val_accuracy = 0.0

    else:
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint_path)
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load epoch
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint['best_val_accuracy']
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load training and validation metrics
        train_losses = checkpoint['train_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_losses = checkpoint['val_losses']
        val_accuracies = checkpoint['val_accuracies']
        
        print(f"Resuming training from epoch {start_epoch}")

    
    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(val_loader, model, criterion, device, epoch)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Update and save training plots
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, '-o', label='Training')
        plt.plot(val_losses, '-o', label='Validation')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, '-o', label='Training')
        plt.plot(val_accuracies, '-o', label='Validation')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'training_validation_plots.png'))
        plt.close()

        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f"best_checkpoint.pth"))

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            }, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--domain', type=str, required=True, help='Name of the domain to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=1, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--resnet_model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18', help='Type of ResNet model to use')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights for ResNet')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    
    main(args)
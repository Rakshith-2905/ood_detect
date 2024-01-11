import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.resnet import CustomResNet
from data_utils.cifar10_data import get_CIFAR10_dataloader
from data_utils.domainnet_data import DomainNetDataset, get_domainnet_loaders
from data_utils.celebA_dataset import get_celebA_dataloader

from train_task_distillation import get_dataset, build_classifier
from data_utils import subpop_bench

def plot_images(loader, title, n_rows=2, n_cols=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Extracts the first batch of images from the given data loader and plots them in a grid with their labels.
    Adjusts the image contrast if necessary.
    """
    # Get the first batch
    if args.dataset_name in subpop_bench.DATASETS:
        _, images, labels, _ = next(iter(loader))
    else:
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

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(args.save_dir, f"{title}.png")
    plt.savefig(save_path)
    # plt.show()

def evaluate(val_loader, model, criterion, device, epoch):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating epoch: {epoch+1}')
    with torch.no_grad():
        for data in pbar:
            if args.dataset_name in subpop_bench.DATASETS:
                inputs, labels = data[1], data[2]
            else:
                inputs, labels = data[0], data[1]

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_loss = loss.item() * inputs.size(0)
            total_loss += batch_loss
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()
            total_correct += batch_correct
            # Set metrics for tqdm
            pbar.set_postfix({"Epoch Loss": total_loss/total_samples, "Epoch Acc": total_correct/total_samples})

    return total_loss/total_samples, total_correct/total_samples

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################### Load Dataset ###########################
    if args.dataset_name == 'domainnet':
        loaders, class_names = get_domainnet_loaders(args.domain, batch_size=args.batch_size, train_shuffle=True)
    elif args.dataset_name in subpop_bench.DATASETS:
        hparams = {
            'batch_size': args.batch_size,
            'image_size': args.image_size,
            'num_workers': 4,
            'group_balanced': None,
        }
        loaders, class_names = subpop_bench.get_dataloader(args.dataset_name, args.data_path, hparams, train_attr='yes')
    elif args.dataset_name == 'CelebA':
        class_attr = 'Young' # attribute for binary classification
        imbalance_attr = ['Male']
        imbalance_percent = {1: [20], 0:[80]} # 1 = Young, 0 = Not Young; 20% of the Young data will be Male
        ignore_attrs = []  # Example: ignore samples that are 'Bald' or 'Wearing_Earrings'

        loaders, class_names = get_celebA_dataloader(args.batch_size, class_attr, imbalance_attr, imbalance_percent, 
                                                     ignore_attrs, img_size=args.image_size, mask=False, mask_region=None)
    elif args.dataset_name == 'cifar10-limited':
        loaders, class_names = get_CIFAR10_dataloader(batch_size=args.batch_size, data_dir=args.data_path, subsample_trainset=True)
    elif args.dataset_name == 'cifar10':
        loaders, class_names = get_CIFAR10_dataloader(batch_size=args.batch_size, data_dir=args.data_path, subsample_trainset=False)
   


    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")

    model, _, _ = build_classifier(args.classifier_model, len(class_names), pretrained=args.use_pretrained)
    model.to(device)

    print(f"Classifier model: {args.classifier_model}")
    print(f"Using pretrained weights: {args.use_pretrained}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Make directory for saving results
    if args.dataset_name == 'domainnet':
        args.save_dir = f"logs/{args.dataset_name}-{args.domain}/{args.classifier_model}/classifier"
    else:
        args.save_dir = f"logs/{args.dataset_name}/{args.classifier_model}/classifier"

    assert os.path.exists(args.save_dir), f"Save directory {args.save_dir} does not exists!"

    plot_images(train_loader, title="Training Image")
    plot_images(val_loader, title="Validation Image")
    plot_images(test_loader, title="Test Image")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Dataparallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)    
    
    val_loss, val_acc = evaluate(test_loader, model, criterion, device, 0)
    print(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}")

    with open(os.path.join(args.save_dir, 'results.txt'), 'w') as f:
        if args.dataset_name == 'domainnet':
            f.write(f"Dataset {args.dataset_name} {args.domain} Test Accuracy: {val_acc:.4f}\n")
        else:
            f.write(f"Dataset {args.dataset_name} Test Accuracy: {val_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train desired classifier model on the desired Dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--domain', type=str, help='Name of the domain if data is from DomainNet dataset')
    parser.add_argument('--data_path', default='./data' ,type=str, help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights for ResNet')
    parser.add_argument('--classifier_model', type=str, choices=['resnet18', 'resnet50', 'vit_b_16', 'swin_b', 'SimpleCNN'], default='resnet18', help='Type of classifier model to use')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    
    main(args)


"""
Sample command to run:
python test_classifier.py \
        --dataset_name cifar10 \
        --data_path ./data \
        --image_size 224 \
        --batch_size 512 \
        --seed 42 \
        --classifier_model SimpleCNN \
        --checkpoint_path logs/cifar10/SimpleCNN/classifier/checkpoint_29.pth


"""
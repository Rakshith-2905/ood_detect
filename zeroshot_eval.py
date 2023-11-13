import torch
import clip
from torchvision.datasets import CIFAR100, ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import os
import logging
import csv

from cifar100_data import get_CIFAR100_loaders, CIFAR100CDataset
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet import CustomResNet, CustomFeatureModel
from models.ViT_models import SAMBackbone, MAEBackbone, DINOBackbone
from models.projector import ProjectionHead
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from ZSL_dataloaders import get_zsl_datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def visualize_dataset(dataset, class_names, num_images=20):
    """ Visualize a grid of images from a dataset with their class names. """
    images, labels = [], []
    for i, (img, label) in enumerate(dataset):
        if i >= num_images:
            break
        images.append(img)
        labels.append(label)

    # Creating a grid of images
    img_grid = make_grid(torch.stack(images, dim=0), nrow=5)

    # Displaying the images and their labels
    plt.figure(figsize=(15, 15))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis('off')

    # Displaying class labels on each sub-image
    for i, label in enumerate(labels):
        print(class_names[label])
        # plt.text(i % 5 * img_grid.size(2) + 20, i // 5 * img_grid.size(1) + 10,
        #          f'Class: {class_names[label]}', color='white', backgroundcolor='black')

    plt.savefig('./gtsrb.png')

def get_transform(feature_extractor_name):
    """ Returns appropriate transform based on model type """
    if feature_extractor_name == 'clip':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    else:  # For projection model
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def build_feature_extractor(feature_extractor_name, feature_extractor_checkpoint_path=None, device="cpu"):
    """
    Builds the feature extractor based on the provided name.
    Args:
        feature_extractor_name (str): The name of the feature extractor to use.
        feature_extractor_checkpoint_path (str, optional): Path to the checkpoint file.
        device (str): The device to load the model onto.
    Returns:
        torch.nn.Module: The feature extractor.
        transforms.Compose: Train transform.
        transforms.Compose: Test transform.
    """
    if feature_extractor_name == 'sam_vit_h':
        if feature_extractor_checkpoint_path is None:
            feature_extractor_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
        feature_extractor = SAMBackbone("vit_h", feature_extractor_checkpoint_path).to(device)
    elif feature_extractor_name == 'mae_vit_large_patch16':
        if feature_extractor_checkpoint_path is None:
            feature_extractor_checkpoint_path = "checkpoints/mae_visualize_vit_large_ganloss.pth"
        feature_extractor = MAEBackbone("mae_vit_large_patch16", feature_extractor_checkpoint_path).to(device)
    elif feature_extractor_name == 'dino_vits16':
        feature_extractor = DINOBackbone("dino_vits16", None).to(device)
    elif feature_extractor_name == 'clip':
        feature_extractor, _ = clip.load(args.clip_model_name, device=device)
        transforms = get_transform('clip')
    elif feature_extractor_name in ['resnet18', 'resnet50', 'resnet101']:
        feature_extractor = CustomResNet(feature_extractor_name, 0, use_pretrained=True).to(device)
    else:
        raise NotImplementedError(f"{feature_extractor_name} is not implemented.")

    train_transform = feature_extractor.transform if hasattr(feature_extractor, 'transform') else None
    test_transform = feature_extractor.test_transform if hasattr(feature_extractor, 'test_transform') else None

    if train_transform is None:
        train_transform = get_transform(feature_extractor_name)
    if test_transform is None:
        test_transform = get_transform(feature_extractor_name)
    
    return feature_extractor, train_transform, test_transform

def load_projector(projector_checkpoint_path, feature_dim, proj_dim, device="cpu"):
    """ Load the projector model """
    projector = ProjectionHead(feature_dim, proj_dim).to(device)
    projector.load_state_dict(torch.load(projector_checkpoint_path, map_location=device)['projector_state'])
    return projector
    
def get_CLIP_text_encodings(texts, device):
    clip_model, _ = clip.load(args.clip_model_name, device=device)

    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_encodings = clip_model.encode_text(text_tokens).float()

    return text_encodings

def get_dataloader(dataset_name, transform, batch_size, device, corruption_type=None, get_text_encodings=False):
    """ Load the appropriate dataset """
    if dataset_name.lower() == 'imagenet':
        dataset = ImageNet(root=args.data_path, download=True, transform=transform, train=False)
        return DataLoader(dataset, batch_size=batch_size)
    if dataset_name.lower() == 'cifar100-c':
        _, class_names = get_zsl_datasets("cifar100", data_path=args.data_path, preprocess=transform)
        if corruption_type:
            dataset = CIFAR100CDataset(data_dir=os.path.join(args.data_path,"CIFAR100-C"), corruption_name=corruption_type, transform=transform)
            print(f"\n\nDataset: {dataset_name}-{corruption_type}\nClasses: {class_names}\nNumber of classes: {len(class_names)}\nNumber of samples: {len(dataset)}")
        else:
            raise ValueError("Corruption type must be specified for CIFAR100-C.")
    elif dataset_name.lower() == 'cifar100':
            _, dataset, class_names = get_CIFAR100_loaders(batch_size=batch_size, data_dir=args.data_path,    
                                            train_transform=transform, test_transform=transform, return_dataset=True)
            dataset = CIFAR100(root=args.data_path, download=True, transform=transform, train=False)
            print(f"\n\nDataset: {dataset_name}\nClasses: {class_names}\nNumber of classes: {len(class_names)}\nNumber of samples: {len(dataset)}")
    elif dataset_name.lower() in ["cifar10", "cifar100", "gtsrb", "svhn", "dtd", "oxfordpets",  "food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        dataset_dict, class_names = get_zsl_datasets(dataset_name, data_path=args.data_path, preprocess=transform)
        dataset = dataset_dict['test']
        #visualize_dataset(dataset, class_names, num_images=20)
        #assert False

        print(f"\n\nDataset: {dataset_name}\nClasses: {class_names}\nNumber of classes: {len(class_names)}\nNumber of samples: {len(dataset)}")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if get_text_encodings:
        text_encodings = get_CLIP_text_encodings(class_names, device)
        return dataLoader, text_encodings


    return dataLoader

def evaluate(model, dataloader, text_encodings, device, feature_extractor_name):
    """ Evaluate the model on the given dataloader """
    
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            if feature_extractor_name == 'clip':
                model.eval()
                features = model.encode_image(images)
            else:

                feature_extractor, projector = model
                feature_extractor.eval()
                projector.eval()

                if feature_extractor_name in ['resnet18', 'resnet50', 'resnet101']:
                    _, backbone_embeddings = feature_extractor(images, return_features=True)
                else:
                    backbone_embeddings = feature_extractor(images)
                features = projector(backbone_embeddings)

            similarities = compute_similarities(features, text_encodings)
    
            probabilities = torch.nn.functional.softmax(similarities, dim=-1)

            total_accuracy += compute_accuracy(probabilities, labels)

    return total_accuracy / len(dataloader)

def save_results_to_csv(dataset_name, feature_extractor_name, accuracy, csv_file="evaluation_results.csv"):
    """ Save evaluation results to a CSV file """
    fieldnames = ['Dataset', 'Feature_Extractor', 'Accuracy']
    row = {'Dataset': dataset_name, 'Feature_Extractor': feature_extractor_name, 'Accuracy': accuracy}
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:  # Write header only if file is empty
            writer.writeheader()
        writer.writerow(row)
    logging.info(f"Results saved to {csv_file}")

def setup_logging():
    """ Set up the logging configuration """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    #supported_datasets = ["cifar10", "cifar100",  "gtsrb", "svhn", "dtd", "oxfordpets",  "food101", "eurosat", "ucf101", "stanfordcars", "flowers102"]
    supported_datasets = ["cifar100"]
    datasets_to_evaluate = supported_datasets if args.dataset_name == 'all' else [args.dataset_name]

    cifar100_corruptions = [
        "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", 
        "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", 
        "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", 
        "snow", "spatter", "speckle_noise", "zoom_blur"
    ]

    # Extract the log path from the projector checkpoint path
    if args.projector_checkpoint_path is not None:
        log_path = os.path.dirname(args.projector_checkpoint_path)
        
        log_path = os.path.join(log_path, 'evaluation')
        try:
            os.makedirs(log_path, exist_ok=True)
        except:
            model_name= args.projector_checkpoint_path.split('/')[-2]
            log_path = os.path.join(os.getcwd(),model_name, 'evaluation')
            os.makedirs(log_path, exist_ok=True)
        logging.info(f"\n\nSaving logs to {log_path}")
        log_file = os.path.join(log_path, f"ZSL_results.csv")
    

    feature_extractor, _, transform = build_feature_extractor(args.feature_extractor_name, args.feature_extractor_checkpoint_path, device)
    if args.feature_extractor_name != 'clip':
        projector = load_projector(args.projector_checkpoint_path, feature_extractor.feature_dim, args.proj_dim, device)
        model = (feature_extractor, projector)
    else:
        model = feature_extractor

    for dataset_name in datasets_to_evaluate:

        if dataset_name not in supported_datasets:
            logging.warning(f"Dataset {dataset_name} not supported. Skipping.")
            continue

        if dataset_name.lower() == 'cifar100-c':
            for corruption in cifar100_corruptions:
                dataloader, text_encodings = get_dataloader(dataset_name, transform, args.batch_size, device, corruption_type=corruption, get_text_encodings=True)
                accuracy = evaluate(model, dataloader, text_encodings, device, args.feature_extractor_name)
                logging.info(f"{args.feature_extractor_name} {dataset_name} {corruption} Accuracy: {accuracy:.6f}")
                save_results_to_csv(f"{dataset_name}-{corruption}", args.feature_extractor_name, accuracy, log_file)
            
        else:
            dataloader, text_encodings = get_dataloader(dataset_name, transform, args.batch_size, device, get_text_encodings=True)
            accuracy = evaluate(model, dataloader, text_encodings, device, args.feature_extractor_name)
            logging.info(f"{args.feature_extractor_name} {dataset_name} Accuracy: {accuracy:.6f}")
            save_results_to_csv(dataset_name, args.feature_extractor_name, accuracy, log_file)
        
        del dataloader, text_encodings

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description='Evaluate CLIP zero-shot performance on a dataset.')
    parser.add_argument('--dataset_name', type=str, nargs='?', const='all', default='all',
                        help='Name of the dataset to evaluate on, or "all" for all datasets.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--text_encoding_path', type=str, required=False)
    parser.add_argument('--feature_extractor_name', type=str, default='clip')  #dino_vits16
    parser.add_argument('--feature_extractor_checkpoint_path', type=str)    
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32')

    parser.add_argument('--projector_checkpoint_path', type=str,default="/p/gpfs1/KDML/ckpts_13M_vivek/dino_vits16full_13M_lr_1e-2_step_lr/projector_weights_epoch_27.pth")
    parser.add_argument('--proj_dim', type=int, default=512)
    args = parser.parse_args()

    print(f"Arguments: {args}")
    #args.data_path =os.path.join("/usr/workspace/thopalli/ICLR2024/rich_datasets", "datasets")
    args.data_path = os.path.join("/usr/workspace/viv41siv/ICASSP2024/ILM_VP_CLIP/datasets")
    args.dataset_name= 'all'
    main(args)

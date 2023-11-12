import torch
import clip
from torchvision.datasets import CIFAR100, ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

from ZSL_dataloaders import get_zsl_datasets

import utils_ddp

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
        feature_extractor, _ = clip.load("RN50", device=device)
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
    projector.load_state_dict(torch.load(projector_checkpoint_path, map_location=device))
    return projector
    
def get_CLIP_text_encodings(texts, device):
    clip_model, _ = clip.load(args.clip_model_name, device=device)

    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_encodings = clip_model.encode_text(text_tokens).float()

    return text_encodings

def get_dataset(dataset_name, transform, batch_size, device, corruption_type=None, get_text_encodings=False):
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
    elif dataset_name.lower() in ["cifar10", "cifar100", "gtsrb", "svhn", "dtd", "oxfordpets",  "food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        dataset_dict, class_names = get_zsl_datasets(dataset_name, data_path=args.data_path, preprocess=transform)
        dataset = dataset_dict['test']
        print(f"\n\nDataset: {dataset_name}\nClasses: {class_names}\nNumber of classes: {len(class_names)}\nNumber of samples: {len(dataset)}")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if get_text_encodings:
        text_encodings = get_CLIP_text_encodings(class_names, device)
        return dataset, text_encodings

    return dataset
    
def reduce_tensor(rt, rank):
    #rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def progbar_wrapper(iterable, total, rank, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if rank == 0:
        return tqdm(iterable, total=total, **kwargs)
    return iterable

def evaluate(model, dataloader, text_encodings, device, feature_extractor_name, rank=0):
    """ Evaluate the model on the given dataloader """
    
    total_accuracy = torch.tensor(0.0).to(device)
    with torch.no_grad():
        pbar = progbar_wrapper(dataloader, total=len(dataloader), rank=rank, desc="Evaluating")
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

    # TODO: CHECK Reduce losses across all processes
    total_accuracy = reduce_tensor(total_accuracy, rank).item()/len(dataloader)

    return total_accuracy
    
def train_one_epoch(train_loader, clip_model, feature_extractor, projector, criterion, optimizer, epoch, rank, device):
    clip_model.eval()
    feature_extractor.eval()
    projector.train()
    
    total_loss = torch.tensor(0.0).to(device)
    total_image_loss = torch.tensor(0.0).to(device)
    total_text_loss = torch.tensor(0.0).to(device)

    pbar = progbar_wrapper(train_loader, total=len(train_loader), rank=rank, desc=f"Training Epoch {epoch + 1}")
    for images_batch, images_clip_batch, captions_batch, image_names_batch in pbar:

        optimizer.zero_grad()
        
        # Ensure data is on the correct device
        images_batch = images_batch.to(device)
        captions_batch = [caption for caption in captions_batch]

        # clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Extract features for images and text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions_batch, truncate=True).to(device)
            clip_txt_embeddings = clip_model.encode_text(text_tokens)

        custom_image_embeddings = feature_extractor(images_batch)

        # Project the resnet embeddings
        proj_embeddings = projector(custom_image_embeddings)

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(clip_txt_embeddings, dim=-1)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # The logits dimension (batch_size, batch_size)
        logits_per_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # 100 is the logits scale from CLIP
        logits_per_text = logits_per_projection.t()

        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries

        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(proj_embeddings)).long().to(device) # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2
        
        loss.backward(loss)

        optimizer.step()

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        if rank == 0:
            pbar.set_postfix({"Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})

    # TODO: CHECK Reduce losses across all processes
    total_loss = reduce_tensor(total_loss, rank).item()/len(train_loader)
    total_image_loss = reduce_tensor(total_image_loss, rank).item()/len(train_loader)
    total_text_loss = reduce_tensor(total_text_loss, rank).item()/len(train_loader)

    return total_loss, total_image_loss, total_text_loss


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

    try:
        utils_ddp.init_distributed_mode_lassen(args)
    except:
        assert Exception("Failed to initialize distributed mode")
    
    world_size = utils_ddp.get_world_size()
    rank = utils_ddp.get_rank()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # supported_datasets = ["cifar10", "cifar100", 'cifar100-c', "gtsrb", "svhn", "dtd", "oxfordpets",  "food101", "eurosat", "ucf101", "stanfordcars", "flowers102"]
    supported_datasets = ["food101", "eurosat", "ucf101", "stanfordcars", "flowers102"]
    datasets_to_evaluate = supported_datasets if args.dataset_name == 'all' else [args.dataset_name]

    cifar100_corruptions = [
        "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", 
        "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", 
        "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", 
        "snow", "spatter", "speckle_noise", "zoom_blur"
    ]

    if rank == 0:

        # Extract the log path from the projector checkpoint path
        if args.projector_checkpoint_path is not None:
            log_path = os.path.dirname(args.projector_checkpoint_path)
            log_path = os.path.join(log_path, 'evaluation')

            os.makedirs(log_path, exist_ok=True)
            logging.info(f"\n\nSaving logs to {log_path}")
            log_file = os.path.join(log_path, f"ZSL_results.csv")
        

    feature_extractor, _, transform = build_feature_extractor(args.feature_extractor_name, args.feature_extractor_checkpoint_path, device)
    feature_extractor = DDP(feature_extractor, device_ids=[0])
    if args.feature_extractor_name != 'clip':
        projector = load_projector(args.projector_checkpoint_path, feature_extractor.feature_dim, args.proj_dim, device)
        projector = DDP(projector, device_ids=[0])
        model = (feature_extractor, projector)
    else:
        model = feature_extractor

    for dataset_name in datasets_to_evaluate:

        if dataset_name not in supported_datasets:
            logging.warning(f"Dataset {dataset_name} not supported. Skipping.")
            continue

        if dataset_name.lower() == 'cifar100-c':
            for corruption in cifar100_corruptions:
                dataset, text_encodings = get_dataloaderset(dataset_name, transform, args.batch_size, device, corruption_type=corruption, get_text_encodings=True)
                                   
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
            
                accuracy = evaluate(model, dataloader, text_encodings, device, args.feature_extractor_name)
                if rank == 0:
                    logging.info(f"{args.feature_extractor_name} {dataset_name} {corruption} Accuracy: {accuracy:.6f}")
                    save_results_to_csv(f"{dataset_name}-{corruption}", args.feature_extractor_name, accuracy, log_file)
        else:
                dataset, text_encodings = get_dataloaderset(dataset_name, transform, args.batch_size, device, get_text_encodings=True)
                                   
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
            
                accuracy = evaluate(model, dataloader, text_encodings, device, args.feature_extractor_name)
                if rank == 0:
                    logging.info(f"{args.feature_extractor_name} {dataset_name} {corruption} Accuracy: {accuracy:.6f}")
                    save_results_to_csv(f"{dataset_name}-{corruption}", args.feature_extractor_name, accuracy, log_file)
        
        del dataloader, text_encodings

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description='Evaluate CLIP zero-shot performance on a dataset.')
    parser.add_argument('--dataset_name', type=str, nargs='?', const='all', default='all',
                        help='Name of the dataset to evaluate on, or "all" for all datasets.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--text_encoding_path', type=str, required=True)
    parser.add_argument('--feature_extractor_name', type=str, default='clip')
    parser.add_argument('--feature_extractor_checkpoint_path', type=str)
    parser.add_argument('--clip_model_name', type=str, default='RN50')

    parser.add_argument('--projector_checkpoint_path', type=str)
    parser.add_argument('--proj_dim', type=int, default=1024)

    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    print(f"Arguments: {args}")
    main(args)

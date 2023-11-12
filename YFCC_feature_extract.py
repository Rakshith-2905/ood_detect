import os
try:
    del os.environ['OMP_PLACES']
    del os.environ['OMP_PROC_BIND']
except:
    pass

import argparse
import json
import os
import requests
import logging

from PIL import Image
from io import BytesIO
from tqdm import tqdm
import pandas as pd
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import clip
import utils_ddp
from torchvision import transforms
import random
from models.ViT_models import SAMBackbone, MAEBackbone, DINOBackbone
from models.resnet import CustomFeatureModel, CustomSegmentationModel
from models.projector import ProjectionHead
import lightning as L
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageTextDataset(Dataset):
    def __init__(self, json_file, data_path, start_index=0, end_index=None, transform=None, transform2=None):
        self.data_path = data_path
        self.transform = transform
        self.transform2 = transform2
        self.df = self._load_json(json_file, start_index, end_index)

    def _load_json(self, json_file, start_index, end_index):
        df = pd.read_json(json_file, lines=True)
        if end_index is not None:
            df = df.iloc[start_index:end_index]
        else:
            df = df#df.iloc[start_index:]

        # Update file paths
       
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['absolute_path']
        caption = row['caption']

        try:
            image = Image.open(image_path).convert('RGB')
            image_trans = self.transform(image) if self.transform else image
            image_trans2 = self.transform2(image) if self.transform2 else image
        except Exception as e:
            print(f"Error with image {image_path}: {e}. Selecting a random replacement.")
            random_idx = random.randint(0, len(self.df) - 1)
            return self.__getitem__(random_idx)

        if self.transform2:
            return image_trans, image_trans2, caption, image_path

        return image_trans, caption, image_path

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable

@torch.no_grad()
def process_data_loader(args, feature_extractor, device, transform, save_path, clip_model_name, feature_extractor_name):
    if fabric.is_global_zero:
        os.makedirs(save_path, exist_ok=True)
    
    # Initialize CLIP
    clip_model, _ = clip.load(args.clip_model_name, device=device)
    clip_model = fabric.to_device(clip_model)

    # Initialize the dataset and data loader
    dataset = ImageTextDataset(args.json_file, args.data_path, start_index=args.start_index, end_index=args.end_index, 
                                transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    data_loader = fabric.setup_dataloaders(data_loader)

    accumulated_data = {
        'image_features': [],
        'text_features': [],
        'filenames': [],
        'captions': []
    }
    processed_images = torch.tensor(0)
    fabric.print(processed_images)
    for images_batch, captions_batch, image_names_batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
        images_batch = images_batch.to(device)
        captions_batch = [caption for caption in captions_batch]
        image_names_batch = [image_name for image_name in image_names_batch]

        image_features = feature_extractor(images_batch)
        text_tokens = fabric.to_device(clip.tokenize(captions_batch, truncate=True))
        text_features = clip_model.encode_text(text_tokens)
        fabric.print(f'text_features.shape: {text_features.shape}')
        fabric.print(f'image_features.shape: {image_features.shape}') 

        # Gather features and filenames
        gathered_image_features = fabric.all_gather(image_features).view(-1, image_features.shape[-1])
        gathered_text_features = fabric.all_gather(text_features).view(-1, text_features.shape[-1])
        gathered_filenames = fabric.all_gather(image_names_batch)
        gathered_captions = fabric.all_gather(captions_batch)

        processed_images += gathered_image_features.shape[0]
        fabric.print(f'gathered_image_features.shape: {gathered_image_features.shape}, gathered_text_features.shape: {gathered_text_features.shape} ')
        fabric.print(f'processed_images: {processed_images}')
      
        accumulated_data['image_features'].append(gathered_image_features)
        accumulated_data['text_features'].append(gathered_text_features)
        accumulated_data['filenames'].extend(gathered_filenames)
        accumulated_data['captions'].extend(gathered_captions)
        
        # Save and reset after processing X number of images
        if processed_images >= args.images_per_chunk:
            save_chunk_data(accumulated_data, save_path, feature_extractor_name, clip_model_name)
            # Reset for the next chunk
            accumulated_data = {
                'image_features': [],
                'text_features': [],
                'filenames': [],
                'captions': []
            }
            processed_images = 0

    # Save any remaining data
    if processed_images > 0:
        save_chunk_data(accumulated_data, save_path, feature_extractor_name, clip_model_name)

    if fabric.is_global_zero:
        logging.info(f"Completed processing all images.")

def save_chunk_data(accumulated_data, save_path, feature_extractor_name, clip_model_name):
    # Convert lists of tensors to a single tensor
    image_features_tensor = torch.cat(accumulated_data['image_features'], dim=0).detach().cpu()
    text_features_tensor = torch.cat(accumulated_data['text_features'], dim=0).detach().cpu()
    print(f'image_features_tensor.shape: {image_features_tensor.shape}, text_features_tensor.shape: {text_features_tensor.shape}')
    
    # Prepare data for saving
    save_data = {
        'image_features': image_features_tensor,
        'text_features': text_features_tensor,
        'filenames': accumulated_data['filenames'],
        'captions': accumulated_data['captions']
    }

    # Construct filename
    save_filename = os.path.join(save_path, f"{feature_extractor_name}_{clip_model_name}_data_chunk.pt")

    # Save the combined data
    torch.save(save_data, save_filename)
    
    assert False


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

def build_feature_extractor(feature_extractor_name, feature_extractor_checkpoint_path=None, device='cpu'):
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
        feature_extractor = SAMBackbone("vit_h", feature_extractor_checkpoint_path)
    elif feature_extractor_name == 'mae_vit_large_patch16':
        if feature_extractor_checkpoint_path is None:
            feature_extractor_checkpoint_path = "checkpoints/mae_visualize_vit_large_ganloss.pth"
        feature_extractor = MAEBackbone("mae_vit_large_patch16", feature_extractor_checkpoint_path)
    elif feature_extractor_name == 'dino_vits16':
        feature_extractor = DINOBackbone("dino_vits16", None)
    elif feature_extractor_name == 'clip':
        feature_extractor, _ = clip.load(args.clip_model_name, device=device)
        transforms = get_transform('clip')
    elif feature_extractor_name in ['resnet18', 'resnet50', 'resnet101']:
        feature_extractor = CustomResNet(feature_extractor_name, 0, use_pretrained=True)
    else:
        raise NotImplementedError(f"{feature_extractor_name} is not implemented.")

    train_transform = feature_extractor.transform if hasattr(feature_extractor, 'transform') else None
    test_transform = feature_extractor.test_transform if hasattr(feature_extractor, 'test_transform') else None

    if train_transform is None:
        train_transform = get_transform(feature_extractor_name)
    if test_transform is None:
        test_transform = get_transform(feature_extractor_name)
    
    return feature_extractor, test_transform, test_transform

def main(args):

    # Initialize the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize the feature extractor
    feature_extractor, transform, _ = build_feature_extractor(args.feature_extractor_name, args.feature_extractor_checkpoint_path, device)
    feature_extractor.to(device)

    process_data_loader(args, feature_extractor, device, transform, args.save_path, args.clip_model_name, args.feature_extractor_name)


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Process a large JSON file and extract image and text features.')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file.')
    parser.add_argument('--feature_extractor_name', required=True, choices=['sam_vit_h', 'mae_vit_large_patch16', 'dino_vits16'],  help='Name of the feature extractor to use.')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--save_path', required=True, help='Path where to save the features.')
    parser.add_argument('--data_path', required=True, help='Path to the data.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for the feature extractor.')
    parser.add_argument('--feature_extractor_checkpoint_path', type=str)
    parser.add_argument('--images_per_chunk', type=int, default=100, help='How many batches make a chunk.')
    parser.add_argument('--start_index', type=int, default=0, help='The starting line index in the JSON file.')
    parser.add_argument('--end_index', type=int, required=True, help='The ending line index in the JSON file.')
    # add num_nodes
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes for distributed training.')
    # add num_gpus
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of gpus per node for distributed training.')
    args = parser.parse_args()
    
    fabric = L.Fabric(accelerator="cuda",num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto")
   
    fabric.launch()
    

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6

    main(args)

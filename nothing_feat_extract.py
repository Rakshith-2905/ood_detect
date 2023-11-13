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
import time
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

@torch.no_grad()
def process_data_loader(args, feature_extractor, device, transform, save_path, clip_model_name, feature_extractor_name):
    # Initialize CLIP
    clip_model, _ = clip.load(args.clip_model_name, device=device)

    # Initialize the dataset and data loader
    dataset = ImageTextDataset(args.json_file, args.data_path, start_index=args.start_index, end_index=args.end_index, 
                                transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    accumulated_data = {
        'image_features': [],
        'text_features': [],
        'filenames': [],
        'captions': []
    }
    processed_images = 0
    chunk_start_index = args.start_index
    for images_batch, captions_batch, image_names_batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
        images_batch = images_batch.to(device)
        captions_batch = [caption for caption in captions_batch]
        image_names_batch = [image_name for image_name in image_names_batch]

        image_features = feature_extractor(images_batch)
        text_tokens = clip.tokenize(captions_batch, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        print(f'text_features.shape: {text_features.shape}')
        print(f'image_features.shape: {image_features.shape}') 

        processed_images += image_features.shape[0]
      
        accumulated_data['image_features'].append(image_features)
        accumulated_data['text_features'].append(text_features)
        accumulated_data['filenames'].extend(image_names_batch)
        accumulated_data['captions'].extend(captions_batch)
        
        # Save and reset after processing X number of images
        if processed_images >= args.images_per_chunk:
            chunk_end_index = chunk_start_index + processed_images - 1
            save_chunk_data(accumulated_data, save_path, chunk_start_index, chunk_end_index, feature_extractor_name, clip_model_name)
            # Reset for the next chunk
            accumulated_data = {
                'image_features': [],
                'text_features': [],
                'filenames': [],
                'captions': []
            }
            processed_images = 0
            chunk_start_index = chunk_end_index + 1

    # Save any remaining data
    if processed_images > 0:

        chunk_end_index = chunk_start_index + processed_images - 1
        save_chunk_data(accumulated_data, save_path, chunk_start_index, chunk_end_index, feature_extractor_name, clip_model_name)
        chunk_start_index = chunk_end_index + 1

    logging.info(f"Completed processing all images.")

def save_chunk_data(accumulated_data, save_path, chunk_start_index, chunk_end_index, feature_extractor_name, clip_model_name):
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

    save_filename = f"{save_path}/{feature_extractor_name}_{chunk_start_index}_{chunk_end_index}.pt"
  
    # Save the combined data
    torch.save(save_data, save_filename)
    logging.info(f"Saved chunk {chunk_start_index} to {chunk_end_index} to {save_filename}")
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
        model, _ = clip.load(args.clip_model_name, device=device)
        feature_extractor = model.encode_image
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
    parser.add_argument('--feature_extractor_name', required=True,    help='Name of the feature extractor to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--save_path', required=True, help='Path where to save the features.')
    parser.add_argument('--data_path', required=True, help='Path to the data.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for the feature extractor.')
    parser.add_argument('--feature_extractor_checkpoint_path', type=str)
    parser.add_argument('--images_per_chunk', type=int, default=100, help='How many batches make a chunk.')
    parser.add_argument('--start_index', type=int, default=0, help='The starting line index in the JSON file.')
    parser.add_argument('--end_index', type=int, required=True, help='The ending line index in the JSON file.')

    args = parser.parse_args()

    # Makes image per chunk a multiple of batch size*world size
    args.images_per_chunk = args.images_per_chunk - (args.images_per_chunk % args.batch_size)
    
    print(f"No. of images per chunk: {args.images_per_chunk}")
    time.sleep(10)
    main(args)

import argparse
import json
import os
import requests
import logging
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
import clip

from models.ViT_models import SAMBackbone, MAEBackbone, DINOBackbone
import lightning as L


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageTextDataset(Dataset):
    def __init__(self, json_file, data_path, start_index=0, end_index=None, transform=None, transform2=None):
        self.data_path = data_path
        self.transform = transform
        self.transform2 = transform2
        self.samples = self._load_json(json_file, start_index, end_index)

    def _load_json(self, json_file, start_index, end_index):
        samples = []
        with open(json_file, 'r') as f:
            for i, line in enumerate(f):
                if i < start_index:
                    continue
                if end_index is not None and i > end_index:
                    break
                data = json.loads(line)
                image_path = os.path.join(self.data_path, data['filename'].split('/')[-1].split('.')[0]+'.jpg')
                if os.path.exists(image_path):
                    samples.append({'image_path': image_path, 'caption': data['caption']})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path'])
        if self.transform:
            image_trans = self.transform(image)
        if self.transform2:
            image_trans2 = self.transform2(image)
            return image_trans, image_trans2, sample['caption'], sample['image_path']
            
        return image_trans, sample['caption'], sample['image_path']

def save_chunk_features(image_features, text_features, save_path, chunk_start_index, chunk_end_index, feature_extractor_name, clip_model_name, chunk_filenames):
    # Combine the features from all batches
    combined_image_features = torch.cat(image_features, dim=0)
    combined_text_features = torch.cat(text_features, dim=0)

    # Save the features
    image_features_filename = os.path.join(save_path, f"{feature_extractor_name}_image_features_{chunk_start_index}_{chunk_end_index}.pt")
    text_features_filename = os.path.join(save_path, f"CLIP_{clip_model_name}_text_features_{chunk_start_index}_{chunk_end_index}.pt")
    torch.save(combined_image_features, image_features_filename)
    torch.save(combined_text_features, text_features_filename)

    # Save the chunk filenames
    chunk_filenames_path = os.path.join(save_path, f"chunk_{chunk_start_index}_{chunk_end_index}_filenames.txt")
    with open(chunk_filenames_path, 'w') as f:
        for filename in chunk_filenames:
            f.write(f"{filename}\n")

def process_data_loader(data_loader, feature_extractor, clip_model, device, save_path, clip_model_name, 
                        feature_extractor_name, num_batches_per_chunk, start_index):
    batch_counter = 0
    processed_images = 0
    chunk_start_index = start_index
    accumulated_image_features = []
    accumulated_text_features = []
    chunk_filenames = []
    
    for images_batch, captions_batch, image_names_batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
        images_batch = images_batch.to(device)
        
        # Extract features for images and text
        with torch.no_grad():
            image_features = feature_extractor(images_batch).detach().cpu()
            text_tokens = clip.tokenize(captions_batch).to(device)
            text_features = clip_model.encode_text(text_tokens).detach().cpu()

        # Accumulate features and filenames
        accumulated_image_features.append(image_features)
        accumulated_text_features.append(text_features)
        chunk_filenames.extend(image_names_batch)
        
        batch_counter += 1
        processed_images += len(images_batch)

        # Save and reset after num_batches_per_chunk
        if batch_counter >= num_batches_per_chunk:
            chunk_end_index = chunk_start_index + processed_images - 1
            save_chunk_features(accumulated_image_features, accumulated_text_features, save_path, 
                                chunk_start_index, chunk_end_index, 
                                feature_extractor_name, clip_model_name, chunk_filenames)
            # Reset for the next chunk
            batch_counter = 0
            accumulated_image_features = []
            accumulated_text_features = []
            chunk_filenames = []
            chunk_start_index = chunk_end_index + 1

    # Save any remaining data that didn't fill a complete chunk
    if batch_counter > 0:
        chunk_end_index = chunk_start_index + processed_images - 1
        save_chunk_features(accumulated_image_features, accumulated_text_features, save_path, 
                            chunk_start_index, chunk_end_index, 
                            feature_extractor_name, clip_model_name, chunk_filenames)

    logging.info(f"Completed processing {processed_images} images.")

def build_feature_extractor(feature_extractor_name):
    """
    Builds the feature extractor based on the provided name.
    Args:
        feature_extractor_name (str): The name of the feature extractor to use.
    Returns:
        torch.nn.Module: The feature extractor.
    """
    if args.feature_extractor_name == 'sam_vit_h':
        feature_extractor = SAMBackbone("vit_h", "checkpoints/sam_vit_h_4b8939.pth").to(device)
    elif args.feature_extractor_name == 'mae_vit_large_patch16':
        feature_extractor = MAEBackbone("mae_vit_large_patch16", "checkpoints/mae_visualize_vit_large_ganloss.pth")
    elif args.feature_extractor_name == 'dino_vits16':
        feature_extractor = DINOBackbone("dino_vits16", None)
    else:
        raise NotImplementedError(f"{feature_extractor_name} is not implemented.")

    transform = feature_extractor.transform
    return feature_extractor, transform

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Process a large JSON file and extract image and text features.')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file.')
    parser.add_argument('--feature_extractor_name', required=True, choices=['sam_vit_h', 'mae_vit_large_patch16', 'dino_vits16'],  help='Name of the feature extractor to use.')
    parser.add_argument('--clip_model_name', default='RN50', help='Name of the CLIP model to use.')
    parser.add_argument('--save_path', required=True, help='Path where to save the features.')
    parser.add_argument('--data_path', required=True, help='Path to the data.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for the feature extractor.')
    parser.add_argument('--num_batches_per_chunk', type=int, default=100, help='How many batches make a chunk.')
    parser.add_argument('--start_index', type=int, default=0, help='The starting line index in the JSON file.')
    parser.add_argument('--end_index', type=int, required=True, help='The ending line index in the JSON file.')

    args = parser.parse_args()

    # Initialize CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model_name, device=device)

    fabric = L.Fabric(accelerator="cuda", devices=8, strategy="ddp")
    fabric.launch()


    feature_extractor, transform = build_feature_extractor(args.feature_extractor_name)
    # Create the dataset and data loader
    dataset = ImageTextDataset(args.json_file, args.data_path, start_index=args.start_index, end_index=args.end_index, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    feature_extractor, optimizer = fabric.setup(feature_extractor, optimizer)
    data_loader = fabric.setup_dataloaders(data_loader)

    # Process the dataset
    process_data_loader(data_loader, feature_extractor, model, device, args.save_path, args.clip_model_name, 
                        args.feature_extractor_name, args.num_batches_per_chunk, args.start_index)
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

from models.ViT_models import SAMBackbone

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_image(data):
    image_path = os.path.join(args.data_path, data['filename'].split('/')[-1].split('.')[0]+'.jpg')
    if not os.path.exists(image_path):
        return None
    image = Image.open(image_path)
    return image

def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f)

def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def get_clip_text_encodings(caption):
    text = clip.tokenize(caption).to(device)
    text_features = model.encode_text(text)
    return text_features

def get_clip_image_features(images_batch):
    # Replace with your own feature extraction process.
    with torch.no_grad():
        image_features = model.encode_image(images_batch)
    return image_features

def save_chunk_features(image_features, text_features, save_path, chunk_start_index, chunk_end_index, feature_extractor_name, clip_model_name, chunk_data):
    """
    Saves the image and text features to disk.
    Args:
        image_features (torch.Tensor): The image features to save.
        text_features (torch.Tensor): The text features to save.
        save_path (str): The path to save the features to.
        chunk_start_index (int): The starting index of the chunk.
        chunk_end_index (int): The ending index of the chunk.
        feature_extractor_name (str): The name of the feature extractor used.
        clip_model_name (str): The name of the CLIP model used.
        chunk_data (dict): The chunk information to save.
    """
    
    # Define the filenames for the chunk
    image_features_filename = f"{feature_extractor_name}_image_features_{chunk_start_index}_{chunk_end_index}.pt"
    text_features_filename = f"CLIP_{clip_model_name}_text_features_{chunk_start_index}_{chunk_end_index}.pt"
    
    # Save the image features
    image_features_path = os.path.join(save_path, image_features_filename)
    try:
        torch.save(image_features, image_features_path)
        logging.info(f"Saved image features to {image_features_path}")
    except Exception as e:
        logging.error(f"Failed to save image features to {image_features_path}: {e}")

    # Save the text features
    text_features_path = os.path.join(save_path, text_features_filename)
    try:
        torch.save(text_features, text_features_path)
        logging.info(f"Saved text features to {text_features_path}")
    except Exception as e:
        logging.error(f"Failed to save text features to {text_features_path}: {e}")

    # Save the chunk data as a JSON file
    chunk_data_filename = f"chunk_data_{chunk_start_index}_{chunk_end_index}.json"
    chunk_data_path = os.path.join(save_path, chunk_data_filename)
    try:
        with open(chunk_data_path, 'w') as f:
            json.dump(chunk_data, f, indent=4)
        logging.info(f"Saved chunk data to {chunk_data_path}")
    except Exception as e:
        logging.error(f"Failed to save chunk data to {chunk_data_path}: {e}")


    # Check if files exist to confirm they were saved
    if os.path.exists(image_features_path) and os.path.exists(text_features_path):
        logging.info(f"Chunk saved successfully: {chunk_start_index} to {chunk_end_index}")
    else:
        logging.error(f"Chunk saving failed for indices {chunk_start_index} to {chunk_end_index}")

def process_images(args, feature_extractor):

    # Create the save path if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        logging.info(f"Created save path: {args.save_path}")
    # checkpoint_file = os.path.join(args.save_path, 'checkpoint.txt')
    # start_index = 0
    # if args.resume and os.path.isfile(checkpoint_file):
    #     with open(checkpoint_file, 'r') as f:
    #         start_index = int(f.read().strip())
    #     logging.info(f"Resuming from line: {start_index}")
    
    # Calculate the starting and ending line indices based on chunks
    start_line = args.start_chunk
    end_line = args.end_chunk

    # Calculate total lines to process for the progress bar
    total_lines = end_line - start_line + 1 if end_line != float('inf') else None

    images_batch = []
    captions_batch = []
    processed_images = 0
    chunk_data = {}

    chunk_start_index = args.start_chunk
    with open(args.json_file, 'r') as f:
        # Initialize the progress bar with the total lines to process
        progress_bar = tqdm(f, total=total_lines, initial=start_line, desc="Processing images", unit="image", dynamic_ncols=True)

        for i, line in enumerate(progress_bar, start_line):
            if i > end_line:
                break  # Stop processing if the end line is reached

            #try:
            data = json.loads(line)
            image = load_image(data)
            if not image:
                logging.error(f"Image not found line {i}")
                continue
            images_batch.append(image)
            captions_batch.append(data['caption'])

            chunk_data[i] = data

            # Update the progress bar description with the current line number
            progress_bar.set_description(f"Processing image line {i}")

            if len(images_batch) == args.batch_size:
                # Process batch
                with torch.no_grad():
                    image_features = feature_extractor(images_batch).detach().cpu()
                    text_features = get_clip_text_encodings(captions_batch).detach().cpu()

                # Accumulate processed features
                if processed_images == 0:
                    chunk_image_features = image_features
                    chunk_text_features = text_features
                else:
                    chunk_image_features = torch.cat((chunk_image_features, image_features), dim=0)
                    chunk_text_features = torch.cat((chunk_text_features, text_features), dim=0)

                processed_images += args.batch_size

                # Reset batch
                images_batch = []
                captions_batch = []

                # Save chunk if it reaches the specified chunk size
                if processed_images % args.chunk_size == 0:
                    chunk_end_index = chunk_start_index + args.chunk_size - 1

                    save_chunk_features(chunk_image_features, chunk_text_features, args.save_path, 
                                        chunk_start_index, chunk_end_index, 
                                        args.feature_extractor_name, args.clip_model_name, chunk_data)
                    
                    # Reset chunk features after saving
                    chunk_image_features = torch.empty(0)
                    chunk_text_features = torch.empty(0)
                    
                    # Reset chunk data after saving
                    chunk_data = {}
                    # Update the chunk start index
                    chunk_start_index = chunk_end_index + 1

            # except Exception as e:
            #     logging.error(f"Error processing line {i}: {e}")

            if i % 100 == 0:
                logging.info(f"Processed {i} lines.")

            # # Save the last processed index after each line
            # with open(checkpoint_file, 'a') as f:
            #     f.write(f'{i}\n')

    # Final save for any remaining data that didn't complete a full chunk
    if images_batch or (processed_images % args.chunk_size != 0):

        if images_batch:
            # Process the final batch if there are any images left
            with torch.no_grad():
                image_features = feature_extractor(images_batch).detach().cpu()
                text_features = get_clip_text_encodings(captions_batch).detach().cpu()
            
            # Concatenate the features of the final batch with those of the chunk
            chunk_image_features = torch.cat((chunk_image_features, image_features), dim=0) if chunk_image_features.numel() else image_features
            chunk_text_features = torch.cat((chunk_text_features, text_features), dim=0) if chunk_text_features.numel() else text_features

        chunk_end_index = chunk_start_index + chunk_image_features.shape[0] - 1

        save_chunk_features(chunk_image_features, chunk_text_features, args.save_path,
                            chunk_start_index, chunk_end_index,
                            args.feature_extractor_name, args.clip_model_name, chunk_data)

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
    else:
        raise NotImplementedError(f"{feature_extractor_name} is not implemented.")
    return feature_extractor

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Process a large JSON file and extract image and text features.')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file.')
    parser.add_argument('--feature_extractor_name', required=True, choices=['sam_vit_h'],  help='Name of the feature extractor to use.')
    parser.add_argument('--clip_model_name', default='RN50', help='Name of the CLIP model to use.')
    parser.add_argument('--save_path', required=True, help='Path where to save the features.')
    parser.add_argument('--data_path', required=True, help='Path to the data.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the feature extractor.')
    parser.add_argument('--chunk_size', type=int, default=100000, help='How many entries to process before saving a chunk.')
    parser.add_argument('--start_chunk', type=int, default=0, help='The starting chunk index.')
    parser.add_argument('--end_chunk', type=int, required=True, help='The ending chunk index.')
    parser.add_argument('--resume', action='store_true', help='Resume from the last checkpoint.')
    args = parser.parse_args()

    # Initialize CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model_name, device=device)

    feature_extractor = build_feature_extractor(args.feature_extractor_name)

    process_images(args, feature_extractor)
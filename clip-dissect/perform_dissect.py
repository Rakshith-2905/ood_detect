import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch.utils.data.dataset import Subset

import argparse
from tqdm import tqdm
from functools import partial
from datetime import datetime

import clip
import csv
from tqdm import tqdm
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from PIL import Image
import itertools
import json

from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from PIL import Image
import numpy as np
import pandas as pd


import utils
import data_utils
import similarity

from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from YFCC_feature_extract import ImageTextDataset
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from train_projection_distill_cont import build_classifier
from collections import OrderedDict

from imagecorruptions import corrupt
import textwrap

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def do_CLIP_dissect(words, d_probe, target_layer, device, domain, save_dir, similarity_fn):

    target_model, target_preprocess = utils.save_activations(clip_name = args.clip_model_name, target_name = 'plumber', target_layers = target_layer, 
                       d_probe = d_probe, concept_set = args.concept_set, batch_size = 32, 
                       device = device, pool_mode='max', save_dir = save_dir, domain=domain,
                       projector_checkpoint_path=args.projector_checkpoint_path,
                       data_dir=args.data_dir)

    # No Transformations are applied to the images
    pil_data = data_utils.get_data(d_probe, domain=domain, data_dir=args.data_dir) 
    
    save_names = utils.get_save_names(clip_name = args.clip_model_name, target_name = 'plumber',
                                  target_layer = target_layer[0], d_probe = d_probe,
                                  concept_set = args.concept_set, pool_mode='max',
                                  save_dir = save_dir)

    target_save_name, clip_save_name, text_save_name, plumber_save_name = save_names

    clip_similarities, plumber_similarities, target_activations, similarity_matrix = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                                text_save_name, plumber_save_name, similarity_fn, device=device)

    return clip_similarities, plumber_similarities, target_activations, pil_data, similarity_matrix, target_model, target_preprocess

def apply_gaussian_noise(image, severity):
    """ Apply Gaussian noise to a PIL image. """
    np_image = np.array(image)
    noise = np.random.normal(0, severity, np_image.shape)
    noisy_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(noisy_image.astype(np.uint8))

def plot_results_with_noise(words, similarities_clip, similarities_plumber, 
                            target_model, text_encodings, pil_data, target_transforms,
                            class_names, K=10, L=5, target_layer=None, save_dir='./results', device="cuda"):
    """
    Plot results for randomly sampled images with varying levels of Gaussian noise.
    """
    
    # Randomly sample 20 indices
    sampled_indices = random.sample(range(len(pil_data)), 20)

    # Different levels of noise severity
    noise_severities = [0, 0.5, 1, 1.5, 2]

    for severity in noise_severities:
        # Create a figure for plotting
        fig, axs = plt.subplots(nrows=20, ncols=2, figsize=[30, 100], gridspec_kw={'width_ratios': [1, 3]})
        
        for i, idx in enumerate(sampled_indices):
            image, label = pil_data[idx]
            # Apply Gaussian noise and transform
            noisy_image = apply_gaussian_noise(image, severity)
            noisy_image_tensor = target_transforms(noisy_image).unsqueeze(0).to(device)

            # Get activations
            activations, logits = utils.get_target_activations(noisy_image_tensor, text_encodings, target_model, target_layer, device)[0]
            mean_activations = torch.mean(activations, dim=0)
            _, top_neurons = torch.topk(mean_activations, K, largest=True)
            
            # Display the image with label on top in the first column
            axs[i, 0].imshow(noisy_image)
            axs[i, 0].set_title(f"Label: {class_names[label]}", fontsize=10)
            axs[i, 0].axis('off')

            # Prepare and display the text for the second column
            concepts_text = ""
            for neuron in top_neurons:
                scores_clip, top_words_clip = torch.topk(similarities_clip[neuron], k=L, largest=True)
                scores_plumber, top_words_plumber = torch.topk(similarities_plumber[neuron], k=L, largest=True)

                clip_words = ', '.join([words[int(id)] for id in top_words_clip])
                plumber_words = ', '.join([words[int(id)] for id in top_words_plumber])

                concepts_text += f"Neuron {neuron.item()}:\tCLIP: {clip_words}\tPLUMBER: {plumber_words}\n\n"
            
            axs[i, 1].text(0.05, 0.5, concepts_text, va='center', fontsize=8)
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust the top space to fit the title if necessary
        plt.savefig(f'{save_dir}_noise_severity_{severity}.png', dpi=300)
        plt.close()

def plot_results_with_noise(words, similarities_clip, similarities_plumber, 
                            target_model, text_encodings, pil_data, target_transforms,
                            class_names, K, L, target_layer, save_dir, device):
    # Create the root directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    text_data_for_json = {}

    # Randomly sample 20 indices
    sampled_indices = random.sample(range(len(pil_data)), 20)
    corruption_name = 'fog'
    # Iterate over noise severities
    for severity in [0, 1, 3, 5]:
        severity_dir = os.path.join(save_dir, f"{corruption_name}/severity_{severity}")
        os.makedirs(severity_dir, exist_ok=True)

        for i, idx in enumerate(sampled_indices):
            image_id = f"image_{i+1}"
            image, label = pil_data[idx]
            if severity == 0:
                noisy_image = image
            else:
                noisy_image = Image.fromarray(corrupt(np.array(image), corruption_name=corruption_name, severity=severity))
            
            image_path = os.path.join(severity_dir, f"{image_id}.png")
            noisy_image.save(image_path)

            # Convert image to tensor and get activations
            image_tensor = target_transforms(noisy_image).unsqueeze(0).to(device)
            activations, logits = utils.get_target_activations(image_tensor, text_encodings, target_model, target_layer, device)

            mean_activations = torch.mean(activations[0], dim=0)
            _, top_neurons = torch.topk(mean_activations, K, largest=True)

            text_info = ''
            for neuron in top_neurons:
                scores_clip, top_words_clip = torch.topk(similarities_clip[neuron], k=L, largest=True)
                scores_plumber, top_words_plumber = torch.topk(similarities_plumber[neuron], k=L, largest=True)

                clip_words = [words[id] for id in top_words_clip.tolist()]
                plumber_words = [words[id] for id in top_words_plumber.tolist()]

                neuron_text = f"Neuron ID: {neuron.item()}, CLIP: {clip_words}, PLUMBER: {plumber_words}\n"
                text_info += neuron_text

                if severity not in text_data_for_json:
                    text_data_for_json[severity] = {}
                
                text_data_for_json[severity][image_id] = {
                    "label": class_names[label],
                    "logits": logits.detach().cpu().numpy().tolist(),
                    "neurons": []
                }

                text_data_for_json[severity][image_id]["neurons"].append({
                    "neuron_id": neuron.item(),
                    "CLIP": clip_words,
                    "PLUMBER": plumber_words
                })

            # # Plotting using Plotly
            # fig = go.Figure()

            # # Column 1: Image and GT Label
            # fig.add_trace(go.Image(z=noisy_image))
            # fig.update_layout(title_text=class_names[label])

            # # Column 2: Text Information
            # fig.add_trace(go.Scatter(x=[0], y=[0], text=[text_info], mode='text'))

            # # Column 3: Logits Visualization
            # logits_values = logits.detach().cpu().numpy().tolist()
            # fig.add_trace(go.Bar(y=logits_values))

            # fig.update_layout(height=600, width=800, title_text="Image Analysis")
            # fig.write_html(os.path.join(severity_dir, f"{image_id}_plot.html"))

    # Convert the nested dictionary to JSON and save
    with open(os.path.join(save_dir, f"{corruption_name}_texts.json"), 'w') as file:
        json.dump(text_data_for_json, file, indent=4)

# def plot_results_with_noise(words, similarities_clip, similarities_plumber, 
#                                            target_model, text_encodings, pil_data, target_transforms,
#                                            class_names, K, L, target_layer, save_dir, device):
#     # Create the root directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Dictionary to store text data for JSON
#     text_data_for_json = {}

#     # Randomly sample 20 indices
#     sampled_indices = random.sample(range(len(pil_data)), 20)
#     corruption_name = 'gaussian_blur'
#     # Iterate over noise severities and save images and texts
#     for severity in [1,3,5]:
#         severity_dir = os.path.join(save_dir, f"{corruption_name}/severity_{severity}")
#         os.makedirs(severity_dir, exist_ok=True)
        
#         for i, idx in enumerate(sampled_indices):
#             image_id = f"image_{i+1}"
#             image, label = pil_data[idx]
#             # Apply the specified corruption to the image
#             noisy_image = Image.fromarray(corrupt(np.array(image), corruption_name=corruption_name, severity=severity))
        
#             image_path = os.path.join(severity_dir, f"{image_id}.png")
#             noisy_image.save(image_path)
            
#             # Convert image to tensor and get activations
#             image_tensor = target_transforms(noisy_image).unsqueeze(0).to(device)
#             activations, logits = utils.get_target_activations(image_tensor, text_encodings, target_model, target_layer, device)

#             mean_activations = torch.mean(activations[0], dim=0)
#             _, top_neurons = torch.topk(mean_activations, K, largest=True)

#             # Create a nested structure for JSON
#             if severity not in text_data_for_json:
#                 text_data_for_json[severity] = {}
            
#             text_data_for_json[severity][image_id] = {
#                 "label": class_names[label],
#                 "logits": logits.detach().cpu().numpy().tolist(),
#                 "neurons": []
#             }
            
#             for neuron in top_neurons:
#                 scores_clip, top_words_clip = torch.topk(similarities_clip[neuron], k=L, largest=True)
#                 scores_plumber, top_words_plumber = torch.topk(similarities_plumber[neuron], k=L, largest=True)

#                 clip_words = [words[id] for id in top_words_clip.tolist()]
#                 plumber_words = [words[id] for id in top_words_plumber.tolist()]

#                 text_data_for_json[severity][image_id]["neurons"].append({
#                     "neuron_id": neuron.item(),
#                     "CLIP": clip_words,
#                     "PLUMBER": plumber_words
#                 })

#     # Convert the nested dictionary to JSON and save
#     with open(os.path.join(save_dir, f"{corruption_name}_texts.json"), 'w') as file:
#         json.dump(text_data_for_json, file, indent=4)

# def plot_top_k_concepts_neurons(words, target_activations, similarities_clip, similarities_plumber, pil_data, 
#                                 class_names, K=10, L=5, target_layer=None, display=False, save_dir='./results'):
    
#     # # Compute mean activations across images and find top K neurons
#     mean_activations = torch.mean(target_activations, dim=0) # 1, num_activations
#     top_vals, top_neurons = torch.topk(mean_activations, k=K, largest=True) # 1, K

#     # Find top L words for each top K neuron and store in a dictionary
#     top_words_dict_clip = {}
#     top_words_dict_plumber = {}

#     for neuron in top_neurons:
#         scores_clip, top_words_clip = torch.topk(similarities_clip[neuron], k=L, largest=True)
#         scores_plumber, top_words_plumber = torch.topk(similarities_plumber[neuron], k=L, largest=True)

#         top_words_dict_clip[int(neuron)] = [(words[w], scores_clip[e].item() )for e, w in enumerate(top_words_clip)]
#         top_words_dict_plumber[int(neuron)] = [(words[w], scores_plumber[e].item() )for e, w in enumerate(top_words_plumber)]

#         print(f"\nNeuron {neuron}: \nCLIP:{top_words_dict_clip[int(neuron)]}\nPLUMBER:{top_words_dict_plumber[int(neuron)]}")


#     _, top_image_ids = torch.topk(target_activations, k=K, dim=0) # top K images for each neuron , top_ids = [K, 512]
    
#     for j, neuron in enumerate(top_neurons):
#         # Get the top L words for the neuron
#         scores_clip, top_words_clip = torch.topk(similarities_clip[neuron], k=L, largest=True)
#         scores_plumber, top_words_plumber = torch.topk(similarities_plumber[neuron], k=L, largest=True)

#         top_words_clip = [words[int(id)] for id in top_words_clip]
#         top_words_plumber = [words[int(id)] for id in top_words_plumber]
        
#         if display:
            
#             # Plotting (Optional based on your requirement)
#             fig, axs = plt.subplots(nrows=1, ncols=K, figsize=[15, 3])
#             fig.suptitle(f"Layer {target_layer} Neuron {int(neuron)}: \n\nCLIP-dissect top {L} words: {top_words_clip}\nPlumber top {L} words: {top_words_plumber}\n")
#             # Iterate over top K images for the neuron
#             for i, top_id in enumerate(top_image_ids[:, neuron]):
#                 im, label = pil_data[top_id]
#                 im = im.resize([375, 375])
#                 axs[i].imshow(im)
#                 axs[i].axis('off')
#                 axs[i].set_title(f"{class_names[label]}")

#         plt.tight_layout()
#         plt.show()
#         plt.savefig(os.path.join({save_dir}, f"layer_{target_layer}_neuron_{int(neuron)}.png"))
#         plt.close()

def main(args):
    
    similarity_fn = similarity.soft_wpmi

    target_layers = ['ln_post']

    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')

    global_save_dir = os.path.dirname(args.projector_checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the CLIP model
    clip_model, _ = clip.load(args.clip_model_name, device=device)
    convert_models_to_fp32(clip_model)

    # text_encodings= torch.load(args.prompt_path)

    for target_layer in target_layers:
        save_dir = os.path.join(global_save_dir, 'activations', target_layer)
        os.makedirs(save_dir, exist_ok=True)

        clip_similarities, plumber_similarities, target_activations, pil_data, _, target_model, target_preprocess = do_CLIP_dissect(
                                                                words, args.dataset_name, [target_layer], device, args.domain_name, save_dir, similarity_fn)
        # Compute clip text encodings for each class
        with torch.no_grad():
            tokenized_class_names = clip.tokenize(pil_data.class_names).to(device)
            text_encodings = clip_model.encode_text(tokenized_class_names).float()
        
        plot_results_with_noise(words, clip_similarities, plumber_similarities, target_model, text_encodings, pil_data, target_preprocess,
                            class_names=pil_data.class_names, K=10, L=5, target_layer=target_layers, save_dir=save_dir, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--projector_checkpoint_path', type=str, help='Path to checkpoint to load the projector from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, default='prompts/cifar3_prompts.pt', help='Path to the prompt file')

    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the results')
    parser.add_argument('--concept_set', type=str, default='prompts/cifar3_attributes.txt', help='Path to the concept set')


    args = parser.parse_args()
    main(args)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import numpy as np

import torch
from torchvision import transforms
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


import clip

from models.plumber import PLUMBER
from train_task_distillation import get_dataset

from utils import save_activations, get_similarity_from_activations

def unnormalize_toPIL(img_tensor, mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]):
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    unnorm_tensor = torch.clamp(img_tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1), 0, 1)
    pil_image = transforms.ToPILImage()(unnorm_tensor)

    return pil_image

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def get_save_dir(args):

    # Get the base dir of the checkpoint path
    if args.checkpoint_path:
        base_dir = os.path.dirname(args.checkpoint_path)
        base_dir = os.path.join(base_dir, 'dissect', args.prefix)
    else:
        att_name = ""
        if args.dataset_name == "NICOpp":
            if args.attributes:
                att_name = "".join([str(att) for att in args.attributes])
                att_name = f"att_{att_name}"
            else:
                att_name = "att_all"
        
        if args.dataset_name == 'domainnet' and args.domain_name:
            att_name = f"{args.domain_name}"

        base_dir = os.path.join(args.save_dir, args.dataset_name, 'shift_detection', att_name, args.prefix, 'clip', 'dissect')  

    return base_dir  

def get_save_names(target_name, concept_set_name, save_dir):
    PM_SUFFIX = {"max":"_max", "avg":"avg"}

    if concept_set_name is None:
        concept_set_name = (args.gt_concept_set.split("/")[-1]).split(".")[0]
    else:
        concept_set_name = (concept_set_name.split("/")[-1]).split(".")[0]
    
    clip_model_name = args.clip_model_name.replace("/", "_")
    target_save_name = f"{save_dir}/features/{clip_model_name}_{target_name}_{PM_SUFFIX[args.pool_mode]}.pt"
    clip_save_name = f"{save_dir}/features/{clip_model_name}_clip_{PM_SUFFIX[args.pool_mode]}.pt"
    text_save_name = f"{save_dir}/features/{clip_model_name}_text_{concept_set_name}_{PM_SUFFIX[args.pool_mode]}.pt"

    log_path = os.path.join(save_dir, concept_set_name)
    return target_save_name, clip_save_name, text_save_name, log_path

def plot_top_k_concepts_neurons(dataset, target_activations, similarities, concept_set,
                                class_names, K, L, target_layer=None, save_dir=None, display=False):

    # # Compute mean activations across images and find top K neurons
    mean_activations = torch.mean(target_activations, dim=0) # 1, num_activations
    top_vals, top_neurons = torch.topk(mean_activations, k=K, largest=True) # 1, K

    _, top_image_ids = torch.topk(target_activations, k=5, dim=0) # top K images for each neuron , top_ids = [K, 512]
    
    # Find top L words for each top K neuron and store in a dictionary
    all_top_words = []
    top_words_dict_clip = {}
    for j, neuron in enumerate(top_neurons):
        # Get the top L words for the neuron
        scores_clip, top_words_clip = torch.topk(similarities[neuron], k=L, largest=True)

        top_words_clip = [concept_set[int(id)] for id in top_words_clip]
        top_words_dict_clip[int(neuron)] = top_words_clip

        all_top_words.extend(top_words_clip)
        if display:
            # Plotting (Optional based on your requirement)
            fig, axs = plt.subplots(nrows=1, ncols=5, figsize=[15, 3])
            fig.suptitle(f"Layer {target_layer} Neuron {int(neuron)}: \n\nCLIP-dissect top {5} words: {top_words_clip[0:5]}\n")

            # Ensure axs is always an array, even when K=1
            axs = np.array(axs).reshape(-1)

            # Iterate over top K images for the neuron
            for i, top_id in enumerate(top_image_ids[:, neuron]):
                _, label, image = dataset[top_id]
                im = unnormalize_toPIL(image)
                im = im.resize([375, 375])
                axs[i].imshow(im)
                axs[i].axis('off')
                axs[i].set_title(f"{class_names[label]}")

            plt.tight_layout()
            plt.savefig(f"{save_dir}/neuron_{int(neuron)}_top_{L}_words.png")

    # Compute unique words and their occurrence frequency
    word_frequency = Counter(all_top_words)

    return word_frequency, all_top_words, top_words_dict_clip

def do_CLIP_dissect(plumber, target_layer, d_probe, concept_set,
                    batch_size, pool_mode, save_names, device):

    save_activations(plumber, target_layer, d_probe, 
                    concept_set, batch_size, pool_mode, save_names, device)

    target_save_name, clip_save_name, text_save_name = save_names

    output = get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, args.similarity_fn, 
                                                    return_target_feats=True, device="cuda")
    
    similarity, target_activations, similarity_matrix = output

    return similarity, target_activations, similarity_matrix

def precision(gt_concept_set, pred_concepts):
    # Calculating precision
    true_positives = 0
    for concept in pred_concepts:
        if concept in gt_concept_set:
            true_positives += 1

    total_predicted_positive = len(pred_concepts)
    precision = true_positives / total_predicted_positive if total_predicted_positive > 0 else 0

    return precision 

def get_text_encodings(clip_model, texts):

    # Compute text embeddings using batch_size=128
    batch_size = 128
    text_encodings = []
    for i in range(0, len(texts), batch_size):
        tokenized_texts = clip.tokenize(texts[i:i+batch_size]).to(args.device)
        text_encodings.append(clip_model.encode_text(tokenized_texts).float().detach().cpu().numpy())
    text_encodings = np.concatenate(text_encodings, axis=0)

    return text_encodings

def plot_UMAP(features_a, features_b, save_dir, name):
    # Plot the UMAP
    import umap
    reducer = umap.UMAP()
    features = np.concatenate((features_a, features_b), axis=0)
    embeddings = reducer.fit_transform(features)
    plt.figure(figsize=(10,10))
    plt.scatter(embeddings[:len(features_a), 0], embeddings[:len(features_a), 1], c='r', label='top retreived')
    plt.scatter(embeddings[len(features_a):, 0], embeddings[len(features_a):, 1], c='b', label='GT concepts')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{name}_umap.png'))
    plt.show()

def plot_tsne(features_a, features_b, save_dir, name):
    # Plot the UMAP
    from sklearn.manifold import TSNE
    reducer = TSNE()
    features = np.concatenate((features_a, features_b), axis=0)
    embeddings = reducer.fit_transform(features)
    plt.figure(figsize=(10,10))
    plt.scatter(embeddings[:len(features_a), 0], embeddings[:len(features_a), 1], c='r', label='top retreived')
    plt.scatter(embeddings[len(features_a):, 0], embeddings[len(features_a):, 1], c='b', label='GT concepts')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{name}_tsne.png'))
    plt.show()

def average_cosine_similarity(features_a, features_b):
    similarity_matrix = cosine_similarity(features_a, features_b)

    # Find the max similarity (closest embedding) for each element in features_a
    max_similarities = np.max(similarity_matrix, axis=1)
    
    assert len(max_similarities) == len(features_a)
    # Calculate the average of these maximum similarities
    average_similarity = np.mean(max_similarities)

    return average_similarity

def plot_and_compute_similarities(text_a, text_b, save_dir, name):

    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.to(args.device)

    # Get the text encodings
    text_encodings_a = get_text_encodings(clip_model, text_a)
    text_encodings_b = get_text_encodings(clip_model, text_b)

    # normalize the text encodings
    text_encodings_a = text_encodings_a / np.linalg.norm(text_encodings_a, axis=1, keepdims=True)
    text_encodings_b = text_encodings_b / np.linalg.norm(text_encodings_b, axis=1, keepdims=True)

    # Plot the UMAP
    # plot_UMAP(text_encodings_a, text_encodings_b, save_dir, name)
    plot_tsne(text_encodings_a, text_encodings_b, save_dir, name)

    # Compute the average cosine similarity
    average_similarity = average_cosine_similarity(text_encodings_a, text_encodings_b)

    # # Save the average similarity
    # with open(os.path.join(save_dir, f'{name}_average_similarity.txt'), 'w') as f:
    #     f.write(str(average_similarity))
    return average_similarity

def main(args):
    
    args.pool_mode = 'avg'
    args.similarity_fn = 'soft_wpmi'

    print(args)

    K=50 # top K neurons
    L=100 # top L words
    

    # Load the gt attribute json file
    with open(args.gt_concept_set, 'r') as f:
        gt_att_concept_json = json.load(f)
    
    # Get all the values from the gt_concept_json into a single list
    gt_att_all_concepts = []
    for att, att_concepts in gt_att_concept_json.items():
        gt_att_all_concepts.extend(att_concepts)

    if args.concept_set:
        with open(args.concept_set, 'r') as f: 
            concept_set = (f.read()).split('\n')
    else:
        concept_set = gt_att_all_concepts


    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name, device=device)

    plumber = PLUMBER(args.clip_model_name, args.num_classes, img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, dataset_txt_prompt=args.dataset_txt_prompt, 
                      is_mlp=args.is_mlp, device=args.device)

    plumber.to(args.device)
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print(f"Checkpoint path {args.checkpoint_path} does not exist")
        else:
            plumber.load_checkpoint(args.checkpoint_path)

    # Create the data loader    # Create the data loader and wrap them with Fabric
    train_transform = None
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, train_transform, 
                                                                                        data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                                        img_size=args.img_size, return_failure_set=True, 
                                                                                        sample_by_attributes=args.attributes, domain_name=args.domain_name)

    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]

    # If train_dataset is more than 30000, then sample 30000 images from it
    if len(train_dataset) > 30000:
        train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(30000)[:30000])

    for target_name in args.target_layers:
        target_save_name, clip_save_name, text_save_name, log_path = get_save_names(target_name, args.concept_set, args.save_dir)

        save_names = [target_save_name, clip_save_name, text_save_name]

        os.makedirs(log_path, exist_ok=True)
        similarities, target_activations, similarity_matrix = do_CLIP_dissect(plumber, target_name, train_dataset, concept_set,
                                                                                        args.batch_size, args.pool_mode, save_names, args.device)
        
        word_frequency, all_top_words, top_words_dict = plot_top_k_concepts_neurons(train_dataset, target_activations, similarities, concept_set,
                                                    class_names, K=K, L=L, target_layer=target_name, 
                                                    save_dir=log_path, display=True)

        # For each of the gt attributes, calculate the precision score with respect to 
        # the top L words from the top K neurons
        score_att = {}
        for att, att_concepts in gt_att_concept_json.items():
            precision_score = precision(att_concepts, all_top_words)
            score_att[att] = precision_score
        
        print(f"Precision score for all attributes: {score_att}")

        # Save the score_att as a json file
        score_att_save_path = f"{log_path}/score_att.json"
        with open(score_att_save_path, 'w') as f:
            json.dump(score_att, f, indent=4)

        # convert word_frequency to json
        word_frequency = dict(word_frequency)

        # Sort the json based on the frequency
        word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)}

        # Save the top 100 words and their frequency as a json file
        words_dump_path = f"{log_path}/{target_name}_top_{K}N_{L}W.json"
        with open(words_dump_path, 'w') as f:
            json.dump(word_frequency, f)
        
        words_dump_path = f"{log_path}/{target_name}_top_{K}N_{L}W.json"
        # Load the json file
        with open(words_dump_path, 'r') as f:
            word_frequency = json.load(f)

        list_retreived_captions = []
        for key, val in word_frequency.items():
            list_retreived_captions.extend([key]*val)
        
        log_path = os.path.join(log_path, 'concept_similarity')
        os.makedirs(log_path, exist_ok=True)
        concept_similarity = {}
        # Load the gt attribute json file
        for att, att_concepts in gt_att_concept_json.items():
            similarity = plot_and_compute_similarities(list_retreived_captions, att_concepts, log_path, att)
            # Convert the similarity to float
            similarity = float(similarity)
            concept_similarity[att] = similarity
        
        # Save the concept_similarity as a json file
        score_att_save_path = f"{log_path}/concept_similarity.json"
        with open(score_att_save_path, 'w') as f:
            json.dump(concept_similarity, f, indent=4)

    print("Done\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    parser.add_argument('--img_projection', action='store_true', help='Whether to use task projection or not')
    parser.add_argument('--txt_projection', action='store_true', help='Whether to use text projection or not')
    parser.add_argument('--img_prompting', action='store_true', help='Whether to use image prompting or not')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to load the step1 model from')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--n_promt_ctx', type=int, default=16, help='Number of learnable prompt token for each cls')
    parser.add_argument('--cls_txt_prompts', action='store_true', help='Whether to use learnable prompts or not')
    parser.add_argument('--dataset_txt_prompt', action='store_true', help='Whether to use dataset level prompts or class level prompts')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--proj_clip', action='store_true', help='Whether to project the clip embeddings or the classifier embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--is_mlp', action='store_true', help='Whether to use MLP projection head or not')
    parser.add_argument('--teacher_temp', type=float, default=0.5, help='Temperature for Dino loss')
    parser.add_argument('--student_temp', type=float, default=1, help='Temperature for Dino loss')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--weight_img_loss', type=float, default=0.5, help='Weight for image loss')
    parser.add_argument('--weight_txt_loss', type=float, default=0.5, help='Weight for text loss')
    
    parser.add_argument('--concept_set', type=str, default=None, help='Path to the concept set')
    parser.add_argument('--gt_concept_set', type=str, default=None, help='Path to the ground truth concept set')
    parser.add_argument('--target_layers', type=str, nargs='+', default=['ln_post'], help='Target layers to use for dissection')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    args.save_dir = get_save_dir(args)
    main(args)


'''
python clip-dissect/dissector.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "domainnet" \
    --gt_concept_set 'clip-dissect/domainnet_domain_concepts.json' \
    --num_classes 345 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --img_projection --txt_projection  \
    --checkpoint_path "logs/domainnet/shift_detection/real/plumber_img_text_proj/_bs_256_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/best_projector_weights.pth" \
    --classifier_name 'resnet18' \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/domainnet/domainnet_CLIP_ViT-B_32_text_embeddings.pth" \
    --save_dir "./logs" \
    --n_promt_ctx 16 \
    --num_epochs 10 \
    --optimizer 'sgd' \
    --learning_rate 0.1 \
    --val_freq 1 \
    --proj_clip \
    --projection_dim 512 \
    --teacher_temp 2 \
    --student_temp 1 \
    --weight_img_loss 1 \
    --weight_txt_loss 1

'''
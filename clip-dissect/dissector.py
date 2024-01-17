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
        if args.attributes:
            att_name = "".join([str(att) for att in args.attributes])
            att_name = f"att_{att_name}"
        else:
            att_name = "att_all"

        base_dir = os.path.join(args.save_dir, args.dataset_name, 'shift_detection', att_name, args.prefix, 'clip', 'dissect')  

    return base_dir  

def get_save_names(target_name, concept_set_name, save_dir):
    PM_SUFFIX = {"max":"_max", "avg":"avg"}
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

# Function to encode text into BERT embeddings
def bert_encode(text, model, tokenizer):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        output = model(input_ids)
    return output[0].mean(dim=1).numpy()

def doc_similarity(gt_concept_lists, pred_concepts):
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # Encode all documents
    predicted_doc_embedding = bert_encode(pred_concepts, model, tokenizer)
    ground_truth_embeddings = [bert_encode(doc, model, tokenizer) for doc in gt_concept_lists]

    # Calculate cosine similarities
    cosine_similarities = [cosine_similarity([predicted_doc_embedding], [gt_embedding])[0][0] for gt_embedding in ground_truth_embeddings]

    print("BERT-based Cosine Similarities:\n", cosine_similarities)

def main(args):
    
    args.pool_mode = 'avg'
    args.similarity_fn = 'soft_wpmi'

    print(args)

    K=50 # top K neurons
    L=100 # top L words
    
    with open(args.concept_set, 'r') as f: 
        concept_set = (f.read()).split('\n')

    attributes = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
    # Load the gt attribute json file
    with open(args.gt_concept_set, 'r') as f:
        att_concept_json = json.load(f)
    
    # Get all the values from the gt_concept_json into a single list
    gt_att_all_concepts = []
    for att in attributes:
        gt_att_all_concepts.append(att_concept_json[att])

    gt_att_concept = []
    # select these attributes
    for att in args.attributes:
        gt_att_concept.extend(att_concept_json[attributes[att]])


    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name, device=device)

    plumber = PLUMBER(args.clip_model_name, args.num_classes, img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, dataset_txt_prompt=args.dataset_txt_prompt, 
                      is_mlp=args.is_mlp, device=args.device)

    plumber.to(args.device)
    if args.checkpoint_path:
        plumber.load_checkpoint(args.checkpoint_path)

    # Create the data loader    # Create the data loader and wrap them with Fabric
    train_transform = None
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, train_transform, 
                                                                                        data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                                        img_size=args.img_size, return_failure_set=True, 
                                                                                        sample_by_attributes=args.attributes)

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


        precision_score = precision(gt_att_concept, all_top_words)

        similarities = doc_similarity(gt_att_all_concepts, all_top_words)
        assert False

        # Save the precision score as a csv file with layer name and precision score
        precision_save_path = f"{log_path}/precision.csv"
        with open(precision_save_path, 'a') as f:
            f.write(f"{target_name},{precision_score}\n")

        # convert word_frequency to json
        word_frequency = dict(word_frequency)

        # Sort the json based on the frequency
        word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)}

        # Save the top 100 words and their frequency as a json file
        words_dump_path = f"{log_path}/{target_name}_top_{K}N_{L}W.json"
        with open(words_dump_path, 'w') as f:
            json.dump(word_frequency, f)
        

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
    
    parser.add_argument('--concept_set', type=str, default='prompts/cifar3_attributes.txt', help='Path to the concept set')
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
    --dataset_name "NICOpp" \
    --attributes 1 2 3 \
    --concept_set "clip-dissect/concepts/niccopp_concepts_att_123.txt" \
    --num_classes 60 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --img_projection --txt_projection  \
    --checkpoint_path "logs/NICOpp/shift_detection/att_123/plumber_img_text_proj/_bs_256_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/best_projector_weights.pth" \
    --classifier_name 'resnet18' \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/NICOpp/NICOpp_CLIP_ViT-B_32_text_embeddings.pth" \
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
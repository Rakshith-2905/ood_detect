import os
import sys
import copy
try:
    del os.environ['OMP_PLACES']
    del os.environ['OMP_PROC_BIND']
except:
    pass

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch.utils.data.dataset import Subset

from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
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
import json

from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier

from models.mapping import TaskMapping, MultiHeadedAttentionSimilarity, MultiHeadedAttention, print_layers
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, CutMix, MyAugMix, find_normalization_parameters
from models.cluster import ClusterCreater
from sklearn.metrics import confusion_matrix
CLIP_LOGIT_SCALE = 100


def entropy(prob):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return -1 * torch.sum(prob * torch.log(prob + 1e-15), axis=-1)

def compute_msp(p):
    msp = torch.max(p, dim=1)[0]
    return msp

def compute_energy(logits, T=1.0):
    return -T*torch.logsumexp(logits/T, dim=1)


def get_score(logits):
    #NOTE: Scores have their sign appropraitely modified to reflect the fact that ID data always has higher scores than OOD data
    print(args.score)
    if args.score == 'msp':
        scores = compute_msp(F.softmax(logits, dim=1))
    elif args.score == 'energy':
        scores = -compute_energy(logits)
    elif args.score == 'pe':
        scores = -entropy(F.softmax(logits, dim=1))
    return scores

def calc_gen_threshold(scores, logits, labels, name='classifier'):
    #NOTE: To be used only with ID data
    scores = scores.cpu().data.numpy()
    probs = F.softmax(logits, dim=1).cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    scores = scores.reshape(-1)
    err = np.argmax(np.array(probs), 1) != np.array(labels)
    thresholds = np.linspace(-40, 40,5000)  # Possible thresholds
    max_loss = 10000
    for t in thresholds:
        l = np.abs(np.mean((scores<t)) - np.mean(err))  #np.abs(
        print(l, t)
        if l < max_loss:
            max_loss = l
            threshold = t

    print('Threshold for {} = {}'.format(name, threshold))
    return threshold

def calc_accuracy_from_scores(scores, threshold):
    idx = (scores<threshold)
    gen_error = (idx.sum())/len(scores)
    gen_accuracy = 1.0-gen_error
    return gen_accuracy, ~idx



def get_save_dir(args):
    if args.resume_checkpoint_path:
        return os.path.dirname(args.resume_checkpoint_path)

    projector_name = "mapper"

    att_name = ""
    if args.dataset_name == "NICOpp":
        if args.attributes:
            att_name = "".join([str(att) for att in args.attributes])
            att_name = f"att_{att_name}"
        else:
            att_name = "att_all"
    
    if args.dataset_name == 'domainnet' and args.domain_name:
        att_name = f"{args.domain_name}"

    save_dir = os.path.join(args.save_dir, args.dataset_name, att_name, projector_name)
    
    save_dir_details = f"{args.prefix}_bs_{args.batch_size}_lr_{args.learning_rate}"

    return os.path.join(save_dir, save_dir_details)

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    return tqdm(iterable, total=total, **kwargs)

@torch.no_grad()
def evaluate_classifier(data_loader, classifier, device='cpu'): 
    
    # Set the model to eval mode
    classifier.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    labels_list, logits_list, probs_list = [], [], []
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Evaluation"
    )
    
    for i, (images_batch, labels, _) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        labels = labels.to(device)
        labels_list.append(labels)

        logits = classifier(images_batch)
        logits_list.append(logits)

        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)
    
    labels_list = torch.cat(labels_list, dim=0)
    logits_list = torch.cat(logits_list, dim=0)
    probs_list = torch.cat(probs_list, dim=0)

    classifier_acc = compute_accuracy(probs_list, labels_list)
    print(f'Classifier Accuracy on {args.dataset_name} = {classifier_acc}')
    print(labels_list.shape, logits_list.shape, probs_list.shape)
    return classifier_acc, labels_list, logits_list, probs_list


@torch.no_grad()
def pim_failure_detection(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, mha, optimizer, epoch): 
    
    # Set the model to eval mode
    pim_model.eval()
    mha.eval()

    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_logits=True, use_cutmix=False)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # Split the similarities into class specific dictionary
        pim_similarities = pim_similarities.to(torch.float32)
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        
        # Compute the pim logits using the multiheaded attention
        pim_logits = mha(pim_similarities_dict)

        loss = F.cross_entropy(pim_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels)
        pim_acc = compute_accuracy(pim_probs, labels)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_task_model_acc = fabric.all_gather(total_task_model_acc).mean() / len(data_loader)
    total_pim_acc = fabric.all_gather(total_pim_acc).mean() / len(data_loader)

    performance_dict = {"total_loss": total_loss, "task_model_acc": total_task_model_acc, "pim_acc": total_pim_acc}

    return performance_dict


def main(args):
    
    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    



    mapper,_, _ = build_classifier(args.classifier_name, num_classes=args.num_classes, pretrained=True, checkpoint_path=None)
    
    cutmix = CutMix(args.cutmix_alpha, args.num_classes)
    pim_model = TaskMapping(task_model=classifier, mapping_model=mapper, 
                              task_layer_name=args.task_layer_name, vlm_dim=args.vlm_dim, 
                              mapping_output_size=mapper.feature_dim, cutmix_fn=cutmix)
    
    ########################### Load the dataset ############################

    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, 
                                                            return_failure_set=True)
    

    if args.dataset_name in ['cifar100']:
        # Merge falure dataset with train dataset
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Number of validation examples: {len(val_loader.dataset)}")
    print(f"Number of test examples: {len(test_loader.dataset)}")


    print(f"Loaded checkpoint from {args.classifier_checkpoint_path}")
    
    if args.method == 'baseline':
        classifier.to(device)
        
        # Evaluating task model
        print('Evaluating on Validation Data')
        val_acc, val_labels_list, val_logits_list, val_probs_list = evaluate_classifier(val_loader, classifier, device=device)
        val_scores = get_score(val_logits_list)
        threshold = calc_gen_threshold(val_scores, val_logits_list, val_labels_list, name='classifier')

        # Just for verification
        estimated_val_acc, val_estimated_success_failure_idx = calc_accuracy_from_scores(val_scores, threshold)

        # Repeating this for test data
        print('Evaluating on Test Data')
        test_acc, test_labels_list, test_logits_list, test_probs_list = evaluate_classifier(test_loader, classifier, device=device)
        test_scores = get_score(test_logits_list)
        estimated_test_acc, test_estimated_success_failure_idx = calc_accuracy_from_scores(test_scores, threshold)

        print(f'Score = {args.score}')
        print(f'True Validation Accuracy = {val_acc}, Estimated Validation Accuracy = {estimated_val_acc}, True Test Accuracy = {test_acc}, Estimated Test Accuracy = {estimated_test_acc}')
        
        val_true_success_failure_idx = torch.argmax(val_probs_list, 1) == val_labels_list
        test_true_success_failure_idx = torch.argmax(test_probs_list, 1) == test_labels_list

        print('Confusion Matrices')
        cm_val = confusion_matrix(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
        cm_test = confusion_matrix(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())

        print('Validation Data')
        print(cm_val)
        print(f'Gen Gap = {torch.abs(val_acc-estimated_val_acc)}')
        print(f'Failure Recall = {cm_val[0,0]/(cm_val[0,0]+cm_val[0,1])}')

        print('Test Data')
        print(cm_test)
        print(f'Gen Gap = {torch.abs(test_acc-estimated_test_acc)}')
        print(f'Failure Recall = {cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])}')

    elif args.method == 'pim':
        class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
        class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"]
        class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

        assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

        num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
        

        mha = MultiHeadedAttentionSimilarity(args.num_classes, num_attributes_per_cls=num_attributes_per_cls, num_heads=1, out_dim=1)

        print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")
        print(f"Built {args.classifier_name} mapper")
        print(f"Built MultiHeadedAttention with {args.num_classes} classes and {num_attributes_per_cls} attributes per class")

        clip_model.to(device)
        classifier.to(device)
        pim_model.to(device)
        mha.to(device)
    
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset')
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    parser.add_argument('--method', type=str, default='baseline', help='Baseline or PIM for failure detection')
    parser.add_argument('--score', type=str, default='msp', help='Failure detection score - msp/energy/pe')
    
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--task_layer_name', type=str, default='model.layer1', help='Name of the layer to use for the task model')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for cutmix')
    parser.add_argument('--augmix_severity', type=int, default=3, help='Severity of the augmix')
    parser.add_argument('--augmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for augmix')
    parser.add_argument('--augmix_prob', type=float, default=0.2, help='Probability of using augmix')
    parser.add_argument('--cutmix_prob', type=float, default=0.2, help='Probability of using cutmix')

    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs before using cutmix')
    parser.add_argument('--discrepancy_weight', type=float, default=1.0, help='Weight to multiply the loss by for samples where the task model is correct and the pim model is incorrect')

    parser.add_argument('--attributes_path', type=str, help='Path to the attributes file')
    parser.add_argument('--attributes_embeddings_path', type=str, help='Path to the attributes embeddings file')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--classifier_dim', type=int, default=None, help='Dimension of the classifier output')

    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--vlm_dim', type=int, default=512, help='Dimension of the VLM embeddings')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of gpus for DDP per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for DDP')


    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # Print the arguments
    print(args)
    sys.stdout.flush()
    
    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    os.makedirs(os.path.join(args.save_dir, 'lightning_logs'), exist_ok=True)
    
    print(f"\nResults will be saved to {args.save_dir}")
    
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    tb_logger = TensorBoardLogger(args.save_dir)
    csv_logger = CSVLogger(args.save_dir, flush_logs_every_n_steps=1)

    seed_everything(args.seed)

    main(args)

'''
Example usage:

python train_mapping_network.py \
--data_dir './data' \
--dataset_name Waterbirds \
--num_classes 2 \
--use_saved_features \
--batch_size 128 \
--img_size 224 \
--seed 42 \
--task_layer_name model.layer1 \
--cutmix_alpha 1.0 \
--warmup_epochs 10 \
--discrepancy_weight 1.0 \
--attributes_path clip-dissect/Waterbirds_concepts.json \
--attributes_embeddings_path data/Waterbirds/Waterbirds_attributes_CLIP_ViT-B_32_text_embeddings.pth \
--classifier_name resnet18 \
--classifier_checkpoint_path logs/Waterbirds/failure_estimation/None/resnet18/classifier/checkpoint_99.pth \
--use_imagenet_pretrained \
--clip_model_name ViT-B/32 \
--prompt_path data/Waterbirds/Waterbirds_CLIP_ViT-B_32_text_embeddings.pth \
--num_epochs 100 \
--optimizer adamw \
--learning_rate 1e-3 \
--val_freq 1 \
--save_dir ./logs \
--prefix '' \
--vlm_dim 512 \
--num_gpus 1 \
--num_nodes 1 


'''
'''
python failure_detection_eval.py \
--data_dir '/usr/workspace/viv41siv/ICLR2024/AMP/data/CIFAR100' \
--dataset_name cifar100 \
--num_classes 100 \
--batch_size 128 \
--img_size 32 \
--seed 42 \
--task_layer_name model.layer1 \
--cutmix_alpha 1.0 \
--warmup_epochs 10 \
--discrepancy_weight 1.0 \
--attributes_path clip-dissect/cifar100_core_concepts.json \
--attributes_embeddings_path data/cifar100/cifar100_attributes_CLIP_ViT-B_32_text_embeddings.pth \
--classifier_name resnet18 \
--classifier_checkpoint_path logs/cifar100/resnet18/classifier/checkpoint_199.pth \
--use_imagenet_pretrained \
--clip_model_name ViT-B/32 \
--prompt_path data/cifar100/cifar100_CLIP_ViT-B_32_text_embeddings.pth \
--num_epochs 100 \
--optimizer adamw \
--learning_rate 1e-3 \
--val_freq 1 \
--save_dir ./logs \
--prefix '' \
--vlm_dim 512 \
--num_gpus 2 \
--num_nodes 1 \
--augmix_prob 0.2 \
--cutmix_prob 0.2 \
--method baseline \
--score msp


'''
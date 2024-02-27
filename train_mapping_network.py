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
CLIP_LOGIT_SCALE = 100

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
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
   
def train_one_epoch(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    augmix,cutmix, clip_model, classifier, pim_model, mha, optimizer, epoch): 

    # Set the models to train mode
    pim_model.train()
    mha.train()

    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        labels_orig = labels.clone()
        augmix_flag =bool(torch.bernoulli(torch.tensor(args.augmix_prob)))
        if epoch >= args.warmup_epochs and augmix_flag :
            images_batch = augmix (images_batch)
        cutmix_flag = bool(torch.bernoulli(torch.tensor(args.cutmix_prob)))
        
        # After certain epoch, select_cutmix with probability 0.5
        if epoch >= args.warmup_epochs and cutmix_flag:
            images_batch, labels = cutmix(images_batch, labels) # Cutmix the images and labels, labels are not one hot encoded anymore

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        optimizer.zero_grad()

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, labels, return_task_logits=True)

        # Cosine similarity between the pim image embeddings and the class_attributes_embeddings
        normalized_pim_image_embeddings = F.normalize(pim_image_embeddings, dim=-1)
        normalized_class_attributes_embeddings = F.normalize(class_attributes_embeddings, dim=-1)

        # convertthe normalized_pim_image_embeddings to data type of normalized_class_attributes_embeddings
        normalized_pim_image_embeddings = normalized_pim_image_embeddings.to(normalized_class_attributes_embeddings.dtype)
        pim_similarities = CLIP_LOGIT_SCALE*(normalized_pim_image_embeddings @ normalized_class_attributes_embeddings.t()) # (batch_size, num_classes*num_attributes_perclass)

        # convert the pim_similarities to the data type of the data type of mha
        pim_similarities = pim_similarities.to(torch.float32)
        # Split the similarities into class specific dictionary
        pim_similarities_dict = {}
        start = 0
        for i, class_prompts in enumerate(class_attribute_prompt_list):
            num_attributes = len(class_prompts)
            pim_similarities_dict[i] = pim_similarities[:, start:start+num_attributes]
            start += num_attributes
        
        # Compute the pim logits using the multiheaded attention
        
        
        pim_logits = mha(pim_similarities_dict)

        loss = F.cross_entropy(pim_logits, labels, reduction = 'none') # Can accomodate the non one hot encoded labels as well # loss is of shape (batch_size,)

        # Check if in this batch if for any samples the task model is correct and the pim model is incorrect
        # if so update a count of such samples
        if (not augmix_flag and not cutmix_flag) and epoch >= args.warmup_epochs:
            task_prediction = torch.argmax(task_model_logits, dim=-1)
            pim_prediction = torch.argmax(pim_logits, dim=-1)
            correct_task_pim_incorrect_idx= torch.where((task_prediction == labels) & (pim_prediction != labels))[0]
            loss[correct_task_pim_incorrect_idx]= loss[correct_task_pim_incorrect_idx]*args.discrepancy_weight # Weight the loss by the number of such samples
            print(f"Correct task, incorrect pim: {len(correct_task_pim_incorrect_idx)}")
        loss= loss.mean()

        fabric.backward(loss)
        
        optimizer.step()

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels_orig)
        pim_acc = compute_accuracy(pim_probs, labels_orig)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_task_model_acc = fabric.all_gather(total_task_model_acc).mean() / len(data_loader)
    total_pim_acc = fabric.all_gather(total_pim_acc).mean() / len(data_loader)

    performance_dict = {"total_loss": total_loss, "task_model_acc %":total_task_model_acc *100., "pim_acc %": total_pim_acc *100.}


    return performance_dict

@torch.no_grad()
def validate(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, mha, optimizer, epoch): 
    
    # Set the model to eval mode
    pim_model.eval()
    mha.eval()

    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Test Epoch {epoch+1}"
    )
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        pim_image_embeddings, task_model_logits, _ = pim_model(images_batch, return_task_logits=True)

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

    performance_dict = {"total_loss": total_loss, "task_model_acc %":total_task_model_acc *100., "pim_acc %": total_pim_acc *100.}

    return performance_dict

def main(args):
    
    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    



    mapper,_, _ = build_classifier(args.classifier_name, num_classes=args.num_classes, pretrained=True, checkpoint_path=None)
    
    cutmix = CutMix(args.cutmix_alpha, args.num_classes)
    pim_model = TaskMapping(task_model=classifier, mapping_model=mapper, 
                              task_layer_name=args.task_layer_name, vlm_dim=args.vlm_dim, 
                              mapping_output_size=mapper.feature_dim, cutmix_fn=cutmix)
    
    # This is for the cross modal attention between PIM and the class attributes
    # mha = MultiHeadedAttention(args.num_classes, in_dim=args.vlm_dim, num_heads=1)

    if not os.path.exists(args.attributes_embeddings_path):
        with open(args.attributes_path, 'r') as f:
            class_attributes_dict = json.load(f)
        
        # For each class, get the attributes clip text embeddings
        class_attributes_embeddings = []
        class_attribute_prompts = []
        for class_names, attributes_dict in class_attributes_dict.items():
            
            # attributes has two lists, one for core attributes and one for non core additional attributes
            #attributes = attributes_dict['core_attributes']
            attributes = attributes_dict
            prompt = [f"This is a photo of {class_names} with {attribute}" for attribute in attributes]
            with torch.no_grad():
                tokenized_prompt = clip.tokenize(prompt).to(args.device)
                class_attributes_embeddings.append(clip_model.encode_text(tokenized_prompt))
                class_attribute_prompts.append(prompt)

        class_attributes_embeddings = torch.cat(class_attributes_embeddings, dim=0) # Shape: [num_classes*num_attributes_perclass, embedding_dim]

        # Make the parent directory if it does not exist
        os.makedirs(os.path.dirname(args.attributes_embeddings_path), exist_ok=True)
        # Save the class attributes embeddings and the class prompts in a single file
        torch.save({"class_attributes_embeddings": class_attributes_embeddings, 
                    "class_attribute_prompts": class_attribute_prompts}, args.attributes_embeddings_path)
        
    else:
        class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
        class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"]
        class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

        assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

    num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
    

    mha = MultiHeadedAttentionSimilarity(args.num_classes, num_attributes_per_cls=num_attributes_per_cls, num_heads=1, out_dim=1)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")
    fabric.print(f"Built {args.classifier_name} mapper")
    fabric.print(f"Built MultiHeadedAttention with {args.num_classes} classes and {num_attributes_per_cls} attributes per class")


    clip_model = fabric.to_device(clip_model)
    fabric.to_device(classifier)
    fabric.to_device(pim_model)
    fabric.to_device(mha)

    ########################### Load the dataset ############################

    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, 
                                                            return_failure_set=True)

    try:
        transform_pipeline =train_dataset.dataset.transform1 #TODO: check this
    except:
        transform_pipeline =train_dataset.transform1
    mean, std = find_normalization_parameters (transform_pipeline)
    augmix = MyAugMix(severity=args.augmix_severity, alpha=args.augmix_alpha, mean=mean, std=std) 
    

    if args.dataset_name in ['cifar100']:
        # Merge falure dataset with train dataset
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    fabric.print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fabric.print(f"Number of training examples: {len(train_loader.dataset)}")
    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")

    ########################### Create the optimizer ############################

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(pim_model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(pim_model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(pim_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise Exception("Invalid optimizer")
    
    # Add the MHA parameters to the optimizer
    optimizer.add_param_group({"params": mha.parameters()})

    # Create the multi step learning rate scheduler

    start_epoch = 0

    state = {"clip_model": clip_model, 
            "classifier": classifier,
            "pim_model": pim_model,
            "mha": mha,
            "optimizer": optimizer, 
            "epoch": start_epoch}

    if args.resume_checkpoint_path:
        state = fabric.load(args.resume_checkpoint_path)
        start_epoch = state["epoch"]

        # Load the model and optimizer
        clip_model = state["clip_model"]
        classifier = state["classifier"]
        pim_model = state["pim_model"]
        mha = state["mha"]
        optimizer = state["optimizer"]

        fabric.print(f"Loaded checkpoint from {args.resume_checkpoint_path} at epoch {start_epoch}")
    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        
        train_performance_dict = train_one_epoch(
                                            train_loader, class_attributes_embeddings, class_attribute_prompts,augmix,
                                            cutmix, clip_model, classifier, pim_model, mha, optimizer, epoch)
        

        if epoch % args.val_freq == 0:
            val_performance_dict = validate( 
                test_loader, class_attributes_embeddings, class_attribute_prompts, 
                                             clip_model, classifier, pim_model, mha, optimizer, epoch)
        
        # Print the losses
        fabric.print(f"Epoch: {epoch+1}/{args.num_epochs} | Train Loss: {train_performance_dict['total_loss']:.4f} | Val Loss: {val_performance_dict['total_loss']:.4f} | Train task model Acc: {train_performance_dict['task_model_acc']:.4f} | Val task model Acc: {val_performance_dict['task_model_acc']:.4f} | Train pim Acc: {train_performance_dict['pim_acc']:.4f} | Val pim Acc: {val_performance_dict['pim_acc']:.4f}")
        # Add train_ to all the keys
        train_performance_dict = {f"train_{key}": value for key, value in train_performance_dict.items()}
             
        # Add test_ to all the keys
        val_performance_dict = {f"test_{key}": value for key, value in val_performance_dict.items()}
        losses_dict = {**train_performance_dict, **val_performance_dict}


        fabric.log_dict(losses_dict, step=epoch)
        
        # Save best model based on validation loss
        if val_performance_dict["test_total_loss"] < best_val_loss:
            
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, "pim_weights_best.pth"), state)
        
        if epoch % 5 == 0:
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, f"pim_weights_{epoch+1}.pth"), state)

    fabric.save(os.path.join(args.save_dir, "pim_weights_final.pth"), state)
    fabric.print(f"Finished training for {args.num_epochs} epochs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset')
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')
    
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

    fabric = L.Fabric(accelerator=args.device,num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto", loggers=[tb_logger, csv_logger])
   
    fabric.launch()

    print = fabric.print

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
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
python train_mapping_network.py \
--data_dir './data' \
--dataset_name cifar100 \
--num_classes 100 \
--batch_size 512 \
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
--num_gpus 1 \
--num_nodes 1 \
--augmix_prob 0.2 \
--cutmix_prob 0.2 


'''
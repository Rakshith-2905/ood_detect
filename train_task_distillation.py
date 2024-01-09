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

from domainnet_data import DomainNetDataset, get_data_from_saved_files
from data_utils import subpop_bench

from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder


def get_dataset(data_name, train_transforms, test_transforms, clip_transform, data_dir='../data', train_attr=None):

    if data_name == 'imagenet':
        train_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_train_examples', transform=train_transforms)
        val_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_val_examples', transform=test_transforms)
        class_names = train_dataset.classes

    elif data_name == 'domainnet':
        train_dataset = DomainNetDataset(root_dir=data_dir, domain=args.domain_name, \
                                        split='train', transform=train_transforms, transform2=clip_transform)
        val_dataset = DomainNetDataset(root_dir=data_dir, domain=args.domain_name, \
                                        split='test', transform=test_transforms, transform2=clip_transform)
        class_names = train_dataset.class_names

    elif data_name == 'cifar10':
        selected_classes = [0,1,2]
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform, selected_classes=selected_classes)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform, selected_classes=selected_classes)

        # class_names = ['airplane', 'automobile', 'bird']
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    elif data_name =="cifar10_full":
        
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform,selected_classes= None)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform,selected_classes= None)
        class_names= train_dataset.class_names

    elif data_name in subpop_bench.DATASETS:
        dataset_class = subpop_bench.get_dataset_class(data_name)
        hparams = {} # TODO: Add hparams need it for CMNIST
        train_dataset = dataset_class(data_dir, 'tr', hparams, clip_transform, train_attr=train_attr)
        val_dataset = dataset_class(data_dir, 'va', hparams, clip_transform)
        test_dataset = dataset_class(data_dir, 'te', hparams, clip_transform)

        class_names = train_dataset.class_names
    
    else:
        raise ValueError(f"Invalid dataset name: {data_name}")
    
    return train_dataset, val_dataset, class_names

def read_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_dataset_from_file(data_name, data_dir='../data'):

    # TODO: Path names

    if data_name == 'imagenet':
        # Load the classifier_embeddings, classifer_logits, clip_features, labels
        train_classifier_data = read_from_pkl(os.path.join(data_dir,"classifier", args.classifier_name, "imagenet-train",f"features_targets_logits.pkl"))
        train_clip_data = read_from_pkl(os.path.join(data_dir,"feature_ext","clip","imagenet-train", f"features_targets.pkl"))
        
        train_classifier_logits = torch.from_numpy(train_classifier_data['logits']).float()
        train_classifier_embeddings =torch.from_numpy( train_classifier_data['features']).float()
        train_classifier_labels = torch.from_numpy(train_classifier_data['targets']).long()
        train_clip_features = torch.from_numpy(train_clip_data['features']).float()
        # Create TensorDatasets
        #print all shapes
        print(f"{train_classifier_logits.shape}, {train_classifier_embeddings.shape}, {train_classifier_labels.shape}, {train_clip_features.shape}")
        train_dataset = TensorDataset(train_classifier_logits, train_classifier_embeddings, train_classifier_labels, train_clip_features)
        
        test_classifier_data = read_from_pkl(os.path.join(data_dir,"classifier", args.classifier_name,"imagenet-val",f"features_targets_logits.pkl"))
        test_clip_data = read_from_pkl(os.path.join(data_dir, "feature_ext","clip","imagenet-val", f"features_targets.pkl"))
        
        test_classifier_logits = torch.from_numpy(test_classifier_data['logits']).float()
        test_classifier_embeddings = torch.from_numpy(test_classifier_data['features']).float()
        test_classifier_labels = torch.from_numpy(test_classifier_data['targets']).long()
        test_clip_features = torch.from_numpy(test_clip_data['features']).float()
        # Create TensorDatasets
        test_dataset = TensorDataset(test_classifier_logits, test_classifier_embeddings, test_classifier_labels, test_clip_features)
        
        with open('data/imagenet_class_name.txt') as f:
            class_names = [line.strip() for line in f]
    elif data_name == 'domainnet':
        train_dataset, test_dataset, class_names = get_data_from_saved_files(data_dir, return_dataset=True)#TODO: check this
    elif data_name == 'cifar10':
        train_dataset, test_dataset, class_names = get_data_from_saved_files(data_dir, return_dataset=True,dataset_name="cifar10")
    elif data_name == 'cifar10_full':
        train_dataset, test_dataset, class_names = get_data_from_saved_files(data_dir, return_dataset=True,dataset_name="cifar10_full")
    return train_dataset, test_dataset, class_names

def get_save_dir(args):
    if args.resume_checkpoint_path:
        return os.path.dirname(args.resume_checkpoint_path)

    projector_name = "plumber" if args.proj_clip else "limber"
    is_proj = False

    if args.img_projection:
        projector_name += "_img"
        is_proj = True
    if args.txt_projection:
        projector_name += "_text"
        is_proj = True

    projector_name += "_proj" if is_proj else ""
    projector_name += "_img_prompt" if args.img_prompting else ""
    projector_name += "_dist_LP" if args.cls_txt_prompts and args.is_dist_prompt else ""
    projector_name += "_LP" if args.cls_txt_prompts and not args.is_dist_prompt else ""

    if not args.save_dir:
        classifier_dir = os.path.dirname(os.path.dirname(args.classifier_checkpoint_path))
        save_dir = os.path.join(classifier_dir, projector_name)
    else:
        save_dir = os.path.join(args.save_dir, args.classifier_name)

    epoch = int(os.path.basename(args.classifier_checkpoint_path).split('_')[-1].split('.')[0])
    save_dir_details = f"{args.prefix}_clsEpoch_{epoch}_bs_{args.batch_size}_lr_{args.learning_rate}"
    save_dir_details += f"_teT_{args.teacher_temp}_sT_{args.student_temp}"
    save_dir_details += f"_imgweight_{args.weight_img_loss}_txtweight_{args.weight_txt_loss}_is_mlp_{args.is_mlp}"

    return os.path.join(save_dir, save_dir_details, 'step_1')

@torch.no_grad()
def get_CLIP_text_encodings(clip_model, texts, save_path=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts)
        text_tokens = fabric.to_device(text_tokens)
        text_encodings = clip_model.encode_text(text_tokens).float()
    # text_encoding_save_path = os.path.join(os.getcwd(), "imagenet_classes_text_encodings.pt")
    torch.save(text_encodings,save_path )
    return text_encodings

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
    
def train_one_epoch(data_loader, clip_model, classifier,
                    img_projector, text_projector, text_encodings_raw, class_prompts,
                    clip_prompted_txt_enc, clip_prompted_img_enc,
                    optimizers_dict, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_train_mode(*models):
        for model in models:
            if model:
                model.train()
    
    def set_zero_grad(*optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.zero_grad()
    
    def optimizer_step(*optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.step()

    set_train_mode(img_projector, text_projector, clip_prompted_txt_enc, clip_prompted_img_enc)


    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        set_zero_grad(optimizers_dict)
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        # Learnable Image Prompting
        if clip_prompted_img_enc:
            clip_image_embeddings = clip_prompted_img_enc(images_clip_batch) # (batch_size, embedding_dim)
        else:
            clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = img_projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = img_projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted_txt_enc:
            text_encodings = clip_prompted_txt_enc(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = clip_prompted_txt_enc(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw
            
        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

        
        fabric.backward(loss)

        optimizer_step(optimizers_dict)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_plumber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_plumber_acc})
    
    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    return total_loss, total_base_model_acc, total_plumber_acc, total_image_loss, total_text_loss

@torch.no_grad()
def validate(data_loader, clip_model, classifier,
                    img_projector, text_projector, text_encodings_raw, class_prompts,
                    clip_prompted_txt_enc, clip_prompted_img_enc, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_eval_mode(*models):
        for model in models:
            if model:
                model.eval()

    set_eval_mode(img_projector, text_projector, clip_prompted_txt_enc, clip_prompted_img_enc)


    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        # Learnable Image Prompting
        if clip_prompted_img_enc:
            clip_image_embeddings = clip_prompted_img_enc(images_clip_batch) # (batch_size, embedding_dim)
        else:
            clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = img_projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = img_projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted_txt_enc:
            text_encodings = clip_prompted_txt_enc(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = clip_prompted_txt_enc(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw
            
        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

        
        fabric.backward(loss)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_plumber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_plumber_acc})
    
    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    return total_loss, total_base_model_acc, total_plumber_acc, total_image_loss, total_text_loss

def train_one_epoch_feat(data_loader, clip_model, classifier,
                    img_projector, text_projector, text_encodings_raw, class_prompts,
                    clip_prompted_txt_enc, clip_prompted_img_enc,
                    optimizers_dict, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_train_mode(*models):
        for model in models:
            if model:
                model.train()
    
    def set_zero_grad(*optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.zero_grad()
    
    def optimizer_step(*optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.step()

    set_train_mode(img_projector, text_projector, clip_prompted_txt_enc, clip_prompted_img_enc)


    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    for classifier_logits, classifier_embeddings, labels, clip_image_embeddings in pbar:


        classifier_logits = fabric.to_device(classifier_logits)
        classifier_embeddings = fabric.to_device(classifier_embeddings)
        labels = fabric.to_device(labels)
        clip_image_embeddings = fabric.to_device(clip_image_embeddings)

        set_zero_grad(optimizers_dict)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)
        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = img_projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = img_projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted_txt_enc:
            text_encodings = clip_prompted_txt_enc(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = text_projector(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)
        # print(normalized_proj_embeddings.shape,normalized_text_encodings.shape, logits_projection.shape,classifier_logits.shape)
        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

  
        fabric.backward(loss)
        
        optimizer_step(optimizers_dict)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_plumber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_plumber_acc})
    
    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean()/ len(data_loader)

    return total_loss, total_base_model_acc, total_plumber_acc, total_image_loss, total_text_loss
    
@torch.no_grad()
def validate_feat(data_loader, clip_model, classifier,
                    img_projector, text_projector, text_encodings_raw, class_prompts,
                    clip_prompted_txt_enc, clip_prompted_img_enc, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_eval_mode(*models):
        for model in models:
            if model:
                model.eval()

    set_eval_mode(img_projector, text_projector, clip_prompted_txt_enc, clip_prompted_img_enc)


    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Validation Epoch {epoch+1}"
    )
    for classifier_logits, classifier_embeddings, labels, clip_image_embeddings in pbar:


        classifier_logits = fabric.to_device(classifier_logits)
        classifier_embeddings = fabric.to_device(classifier_embeddings)
        labels = fabric.to_device(labels)
        clip_image_embeddings = fabric.to_device(clip_image_embeddings)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)
        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = img_projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = img_projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted_txt_enc:
            text_encodings = clip_prompted_txt_enc(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = text_projector(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)
        # print(normalized_proj_embeddings.shape,normalized_text_encodings.shape, logits_projection.shape,classifier_logits.shape)
        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

  
        fabric.backward(loss)
        
        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_plumber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_plumber_acc})
    
    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean()/ len(data_loader)

    return total_loss, total_base_model_acc, total_plumber_acc, total_image_loss, total_text_loss
    
def build_classifier(classifier_name, num_classes, pretrained=False, checkpoint_path=None):
    # TODO: Verify each of the models and the transforms
    if classifier_name in ['vit_b_16', 'swin_b']:
        classifier = CustomClassifier(classifier_name, use_pretrained=pretrained)
    elif classifier_name in ['resnet18', 'resnet50']:
        classifier = CustomResNet(classifier_name, num_classes=num_classes, use_pretrained=pretrained)
    elif classifier_name == 'SimpleCNN':
        classifier = SimpleCNN()
    elif classifier_name == 'Resnet18_cifar_10':
        classifier = ResNet18()
        classifier.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        classifier.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if checkpoint_path:

        if classifier_name == 'SimpleCNN':
            classifier.load_state_dict(torch.load(checkpoint_path))
        elif classifier_name == 'Resnet18_cifar_10':
            classifier = torch.nn.DataParallel(classifier)

            classifier.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
            classifier = classifier.module   
        else:
            classifier.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
            fabric.print(f"Loaded {classifier_name} from {checkpoint_path}")
    train_transform = classifier.train_transform
    test_transform = classifier.test_transform

    return classifier, train_transform, test_transform

def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    img_projector = None
    if args.img_projection:
        if args.proj_clip:
            # This is PLUMBER
            img_projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
            fabric.print(f"Constructed img emb projection PLUMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
        else:
            # This is LIMBER
            img_projector = ProjectionHead(input_dim=classifier.feature_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
            fabric.print(f"Constructed img emb projection LIMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
    
    text_projector = None
    if args.txt_projection:
        text_projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
        fabric.print(f"Constructed text emb projection PLUMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")

    ########################### Load the dataset ############################

    if args.use_saved_features:
        # data_dir = os.path.join(args.data_dir, args.domain_name)
        # Get the directory of classifier checkpoints
        data_dir = os.path.dirname(args.classifier_checkpoint_path)
        data_dir = os.path.join(data_dir, 'features', args.domain_name)
        train_dataset, val_dataset, class_names = get_dataset_from_file(args.dataset_name, data_dir=data_dir)
        fabric.print(f"Using saved features from {args.dataset_name} dataset from {data_dir}")
    else:
        # Create the data loader and wrap them with Fabric
        train_dataset, val_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                                data_dir=args.data_dir, clip_transform=clip_transform)
        fabric.print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # TODO: For any new datasets ensure this is handeled properly
    if args.train_on_testset:
        train_loader = val_loader
        fabric.print("Training on test set")

    fabric.print(f"Number of training examples: {len(train_loader.dataset)}")
    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    # Get the text encodings for the class names
    try:
        text_encodings= torch.load(args.prompt_path)
        if text_encodings.shape[0] != len(class_names):
            raise Exception("Text encodings shape does not match the number of classes")

        fabric.print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        fabric.print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    class_prompts = None
    clip_prompted_txt_enc = None
    if args.cls_txt_prompts:

        # Create the prompted CLIP model
        clip_prompted_txt_enc = PromptedCLIPTextEncoder(clip_model, n_ctx=args.n_promt_ctx, num_classes=len(class_names), 
                                                    device=fabric.device, is_dist_prompt=args.is_dist_prompt)
        clip_prompted_txt_enc = fabric.to_device(clip_prompted_txt_enc)

        class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]

        if args.dataset_prompt:
            fabric.print(f"Constructed CLIP Prompted Text Encoder with {len(class_names)} classes")
        else:
            fabric.print(f"Constructed CLIP Prompted Text Encoder- dist prompting")
    
    if args.img_prompting:
        # Create the prompted CLIP model
        clip_prompted_img_enc = PromptedCLIPImageEncoder(clip_model, num_tokens=args.n_promt_ctx, device=fabric.device)
        clip_prompted_img_enc = fabric.to_device(clip_prompted_img_enc)

        fabric.print(f"Constructed CLIP Prompted Image Encoder")

    ########################### Create the optimizer ############################
    optimizer_img_proj = None
    if args.img_projection:
        # Create the optimizer and scheduler
        if args.optimizer == 'adam':
            optimizer_img_proj = torch.optim.Adam(img_projector.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer_img_proj = torch.optim.SGD(img_projector.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer == 'adamw':
            optimizer_img_proj = torch.optim.AdamW(img_projector.parameters(), lr=args.learning_rate)

        img_projector, optimizer_img_proj = fabric.setup(img_projector, optimizer_img_proj)
        scheduler_img_proj = torch.optim.lr_scheduler.MultiStepLR(optimizer_img_proj, milestones=[30, 60, 90], gamma=0.1)

    # Create optimizer for text projection
    optimizer_txt_proj = None
    if args.txt_projection:
        if args.optimizer == 'adam':
            optimizer_txt_proj = torch.optim.Adam(text_projector.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer_txt_proj = torch.optim.SGD(text_projector.parameters(), lr=args.learning_rate, momentum=0.9)
        elif args.optimizer == 'adamw':
            optimizer_txt_proj = torch.optim.AdamW(text_projector.parameters(), lr=args.learning_rate)

        text_projector, optimizer_txt_proj = fabric.setup(text_projector, optimizer_txt_proj)
        scheduler_txt_proj = torch.optim.lr_scheduler.MultiStepLR(optimizer_txt_proj, milestones=[30, 60, 90], gamma=0.1)

    # add clip_prompted to the optimizer
    optimizer_txt_prompt = None
    if args.cls_txt_prompts:
        optimizer_txt_prompt = torch.optim.SGD([p for p in clip_prompted_txt_enc.parameters() if p.requires_grad], lr=0.1)
        clip_prompted_txt_enc, optimizer_txt_prompt = fabric.setup(clip_prompted_txt_enc, optimizer_txt_prompt)
    
    optimizer_img_prompt = None
    if args.img_prompting:
        optimizer_img_prompt = torch.optim.SGD([p for p in clip_prompted_img_enc.parameters() if p.requires_grad], lr=0.1)
        clip_prompted_img_enc, optimizer_img_prompt = fabric.setup(clip_prompted_img_enc, optimizer_img_prompt)
    
    optimizers_dict = {
        "optimizer_img_proj": optimizer_img_proj,
        "optimizer_txt_proj": optimizer_txt_proj,
        "optimizer_txt_prompt": optimizer_txt_prompt,
        "optimizer_img_prompt": optimizer_img_prompt
    }
    scheduler_dict = {
        "scheduler_img_proj": scheduler_img_proj,
        "scheduler_txt_proj": scheduler_txt_proj
    }
        

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
        classifier = fabric.to_device(classifier)
    
    start_epoch = 0

    state = {"clip_model": clip_model, "classifier": classifier, "optimizer_img_proj": optimizer_img_proj,
                "optimizer_txt_proj": optimizer_txt_proj, "optimizer_txt_prompt": optimizer_txt_prompt, 
                "optimizer_img_prompt": optimizer_img_prompt, "epoch": start_epoch}

    if args.resume_checkpoint_path:
        fabric.load(args.resume_checkpoint_path, state)
        start_epoch = state["epoch"] + 1

    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        if args.use_saved_features:
            train_loss,  train_base_acc, train_plumber_acc, train_loss_img, train_loss_txt = train_one_epoch_feat(
                                                                            train_loader, clip_model, classifier,
                                                                            img_projector, text_projector,
                                                                            text_encodings, class_prompts,
                                                                            clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                            optimizers_dict, criterion, epoch)
            if epoch % args.val_freq == 0:
                val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate_feat( clip_model, classifier,
                                                                            img_projector, text_projector,
                                                                            text_encodings, class_prompts,
                                                                            clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                            criterion, epoch)
        else:    
            train_loss,  train_base_acc, train_plumber_acc, train_loss_img, train_loss_txt = train_one_epoch(
                                                                            train_loader, clip_model, classifier,
                                                                            img_projector, text_projector,
                                                                            text_encodings, class_prompts,
                                                                            clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                            optimizers_dict, criterion, epoch)
            if epoch % args.val_freq == 0:
                val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate(
                                                                            train_loader, clip_model, classifier,
                                                                            img_projector, text_projector,
                                                                            text_encodings, class_prompts,
                                                                            clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                            criterion, epoch)
        
        if args.img_projection:
            scheduler_img_proj.step()
        if args.txt_projection:
            scheduler_txt_proj.step()
            

        
        fabric.print(f"Epoch {epoch}/{args.num_epochs}| Train Loss: {train_loss:.4f}, Train Base Model Accuracy: {train_base_acc:.4f}, Train PLUMBER Accuracy: {train_plumber_acc:.4f}, Val Loss: {val_loss:.4f}, Val Base Model Accuracy: {val_base_acc:.4f}, Val PLUMBER Accuracy: {val_plumber_acc:.4f}")

        losses_dict = {"train_loss": train_loss, "train_loss_img": train_loss_img, "train_loss_txt": train_loss_txt,
                        "train_base_acc": train_base_acc, "train_plumber_acc": train_plumber_acc,
                        "val_loss": val_loss, "val_loss_img": val_loss_img, "val_loss_txt": val_loss_txt, 
                        "val_base_acc": val_base_acc, "val_plumber_acc": val_plumber_acc}

        fabric.log_dict(losses_dict, step=epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, "best_projector_weights.pth"), state)
        
        if epoch % 5 == 0:
            state.update(epoch=epoch)
            fabric.save(os.path.join(args.save_dir, f"projector_weights_{epoch+1}.pth"), state)

    fabric.save(os.path.join(args.save_dir, "projector_weights_final.pth"), state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--train_on_testset', action='store_true', help='Whether to train on the test set or not')
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    parser.add_argument('--img_projection', action='store_true', help='Whether to use task projection or not')
    parser.add_argument('--txt_projection', action='store_true', help='Whether to use text projection or not')
    parser.add_argument('--img_prompting', action='store_true', help='Whether to use image prompting or not')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--n_promt_ctx', type=int, default=16, help='Number of learnable prompt token for each cls')
    parser.add_argument('--cls_txt_prompts', action='store_true', help='Whether to use learnable prompts or not')
    parser.add_argument('--dataset_prompt', action='store_true', help='Whether to use dataset level prompts or class level prompts')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--proj_clip', action='store_true', help='Whether to project the clip embeddings or the classifier embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--is_mlp', action='store_true', help='Whether to use MLP projection head or not')
    parser.add_argument('--teacher_temp', type=float, default=0.5, help='Temperature for Dino loss')
    parser.add_argument('--student_temp', type=float, default=1, help='Temperature for Dino loss')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--weight_img_loss', type=float, default=0.5, help='Weight for image loss')
    parser.add_argument('--weight_txt_loss', type=float, default=0.5, help='Weight for text loss')
    
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of gpus for DDP per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for DDP')
    parser.add_argument('--template_num', type=int, default=0, help='CLIP text prompt number')

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

    fabric = L.Fabric(accelerator="cuda",num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto", loggers=[tb_logger, csv_logger])
   
    fabric.launch()

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
            
    seed_everything(args.seed)

    main(args)

'''
 python task_distillation.py \
        --data_dir './data/'  \
        --domain_name 'real'    \
        --dataset_name 'Waterbirds'    \
        --train_on_testset    \
        --num_classes 2  \
        --batch_size 256  \
        --seed 42    \
        --img_projection \
        --cls_txt_prompts \
        --cls_txt_prompts \
        --classifier_name 'resnet18' \
        --classifier_checkpoint_path 'logs/Waterbirds/resnet18/classifier/checkpoint_99.pth' \
        --clip_model_name 'ViT-B/32' \
        --prompt_path 'data/waterbirds/waterbirds_CLIP_ViT-B_32_text_embeddings.pth' \
        --n_promt_ctx 16 \
        --num_epochs 10 \
        --optimizer 'sgd' \
        --learning_rate 0.1 \
        --val_freq 1 \
        --prefix '' \
        --proj_clip \
        --projection_dim 512 \
        --teacher_temp 2.0  \
        --student_temp 1 \
        --weight_img_loss 1.0 \
        --weight_txt_loss 1.0 \
        --num_gpus 1 \
        --num_nodes 1

'''
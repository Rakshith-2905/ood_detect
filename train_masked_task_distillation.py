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
from torchvision.transforms import Normalize, Resize
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

from data_utils.ZSL_dataloaders.prepare_data import get_zsl_datasets, prepare_expansive_data, prepare_gtsrb_fraction_data

from data_utils.domainnet_data import DomainNetDataset, get_data_from_saved_files, get_domainnet_loaders
from data_utils.cifar100_data import CIFAR100C, CIFAR100TwoTransforms, get_CIFAR100_dataloader, get_CIFAR100C_dataloader
from data_utils.cifar10_data import CIFAR10C, CIFAR10TwoTransforms, get_CIFAR10_dataloader, get_CIFAR10C_dataloader
from data_utils.celebA_dataset import FilteredCelebADataset, get_celebA_datatransforms
from data_utils import subpop_bench
from data_utils.imagenet_dataset import ImageNetTwoTransforms, get_imagenet_loaders

from models.resnet import CustomClassifier, CustomResNet, CustomFeatureModel
from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from models.custom_clip import MaskedTextTransformer, MaskedVisualTransformer

from train_task_distillation import build_classifier, get_CLIP_text_encodings, get_dataset
from utils_proj import ImageTransforms

def read_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_save_dir(args):
    if args.resume_checkpoint_path:
        return os.path.dirname(args.resume_checkpoint_path)

    projector_name = "masked_img_encoder" 

    # Add the layers to mask to the projector name if any
    if args.layers_to_mask:
        projector_name += "_layer"
        for layer in args.layers_to_mask:
            projector_name += f"_{layer}"

    if not args.save_dir:
        classifier_dir = os.path.dirname(os.path.dirname(args.classifier_checkpoint_path))
        save_dir = os.path.join(classifier_dir, projector_name)
    else:
        save_dir = os.path.join(args.save_dir, args.classifier_name)

    epoch = int(os.path.basename(args.classifier_checkpoint_path).split('_')[-1].split('.')[0])
    save_dir_details = f"{args.prefix}_clsEpoch_{epoch}_bs_{args.batch_size}_lr_{args.learning_rate}"
    save_dir_details += f"_teT_{args.teacher_temp}_sT_{args.student_temp}"
    save_dir_details += f"_imgweight_{args.weight_img_loss}_txtweight_{args.weight_txt_loss}_regweight_{args.regularization_weight}"

    return os.path.join(save_dir, save_dir_details, 'step_1')

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable

def coral_loss_compute(x, y):

    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff

def train_one_epoch(data_loader, clip_model, classifier,
                    masked_visual_encoder, text_encodings,
                    optimizers_dict, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_train_mode(*models):
        for model in models:
            if model:
                model.train()
    
    def set_zero_grad(optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.zero_grad()
    
    def optimizer_step(optimizers_dict):
        for optimizer_name, optimizer in optimizers_dict.items():
            if optimizer:
                optimizer.step()

    set_train_mode(masked_visual_encoder)


    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        num_in_images = len(images_batch[0])

        images_batch = torch.cat(images_batch, dim=0)
        images_clip_batch = torch.cat(images_clip_batch, dim=0)
        labels = torch.cat(labels, dim=0)
        
        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        set_zero_grad(optimizers_dict)
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        clip_image_embeddings = masked_visual_encoder(images_clip_batch) 

        orig_clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)
            
        normalized_image_embeddings = F.normalize(clip_image_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_image_embeddings)
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_image_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())

        loss_ce = F.cross_entropy(logits_projection, torch.argmax(classifier_logits, dim=-1))

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        # Split the logits into in and out logits
        proj_probs_in = probs_from_proj[:num_in_images]
        proj_probs_out = probs_from_proj[num_in_images:]

        margin_loss = 0.1*(torch.pow(F.relu(proj_probs_in-0.7), 2).mean() + 
                     torch.pow(F.relu(0.4-proj_probs_out), 2).mean())
        
        coral_loss = 0.2*coral_loss_compute(clip_image_embeddings, orig_clip_image_embeddings)

        # Add regularization loss
        regularization_loss = 0
        for name, param in masked_visual_encoder.named_parameters():
            if 'mask' in name:
                regularization_loss += torch.sum(torch.abs(param))

        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss + args.regularization_weight*regularization_loss + 0.5*loss_ce + margin_loss + coral_loss 

        fabric.backward(loss)

        optimizer_step(optimizers_dict)


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
                    masked_visual_encoder, text_encodings,
                    optimizers_dict, criterion, epoch):

    clip_model.eval()
    classifier.eval()
    masked_visual_encoder.eval()



    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Validating Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        clip_image_embeddings = masked_visual_encoder(images_clip_batch) 

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)
            
        normalized_image_embeddings = F.normalize(clip_image_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_image_embeddings)
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_image_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())

        # Add regularization loss
        regularization_loss = 0
        for name, param in masked_visual_encoder.named_parameters():
            if 'mask' in name:
                regularization_loss += torch.sum(torch.abs(param))

        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss + args.regularization_weight*regularization_loss

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

def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)

    # Get the normalizations from the clip_transform
    normalize = clip_transform.transforms[1]
    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    ########################### Load the dataset ############################


    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, return_failure_set=True)

    fabric.print(f"Using {args.dataset_name} dataset")


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    # TODO: For any new datasets ensure this is handeled properly
    if args.train_on_testset:
        train_loader = val_loader
        fabric.print("Training on test set")

    fabric.print(f"Number of training examples: {len(train_loader.dataset)}")
    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")
    # Get the text encodings for the class names
    try:
        text_encodings= torch.load(args.prompt_path)
        if text_encodings.shape[0] != len(class_names):
            raise Exception("Text encodings shape does not match the number of classes")

        fabric.print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        fabric.print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    visual_mask_layers = args.layers_to_mask
    masked_visual_encoder = MaskedVisualTransformer(clip_model, visual_mask_layers)

    ########################### Create the optimizer ############################

    train_params = [p for p in masked_visual_encoder.parameters() if p.requires_grad]
    
    # Create the optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer_img_proj = torch.optim.Adam(train_params, lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer_img_proj = torch.optim.SGD(train_params, lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adamw':
            optimizer_img_proj = torch.optim.AdamW(train_params, lr=args.learning_rate)

    masked_visual_encoder, optimizer_img_proj = fabric.setup(masked_visual_encoder, optimizer_img_proj)
    scheduler_img_proj = torch.optim.lr_scheduler.MultiStepLR(optimizer_img_proj, milestones=[30, 60, 90], gamma=0.1)
    
    optimizers_dict = {
        "optimizer_img_proj": optimizer_img_proj
    }
    
    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
        classifier = fabric.to_device(classifier)
    
    start_epoch = 0

    state = {"clip_model": clip_model, "classifier": classifier, 
            "masked_visual_encoder": masked_visual_encoder,
            "optimizer_img_proj": optimizer_img_proj, "epoch": start_epoch}

    if args.resume_checkpoint_path:
        fabric.load(args.resume_checkpoint_path, state)
        start_epoch = state["epoch"] + 1
        fabric.print(f"Resuming training from epoch {start_epoch}")
        # update the optimizers 

    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return
    
    val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate(
                                                                test_loader, clip_model, classifier,
                                                                masked_visual_encoder, text_encodings,
                                                                optimizers_dict, criterion, -1)
    print(f"Val Loss: {val_loss:.4f}, Val Base Model Accuracy: {val_base_acc:.4f}, Val PLUMBER Accuracy: {val_plumber_acc:.4f}")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
             
        train_loss,  train_base_acc, train_plumber_acc, train_loss_img, train_loss_txt = train_one_epoch(
                                                                        train_loader, clip_model, classifier,
                                                                        masked_visual_encoder, text_encodings,
                                                                        optimizers_dict, criterion, epoch)
        if epoch % args.val_freq == 0:
            val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate(
                                                                        test_loader, clip_model, classifier,
                                                                        masked_visual_encoder, text_encodings,
                                                                        optimizers_dict, criterion, epoch)
        
            
        fabric.print(f"Epoch {epoch+1}/{args.num_epochs}| Train Loss: {train_loss:.4f}, Train Base Model Accuracy: {train_base_acc:.4f}, Train PLUMBER Accuracy: {train_plumber_acc:.4f}, Val Loss: {val_loss:.4f}, Val Base Model Accuracy: {val_base_acc:.4f}, Val PLUMBER Accuracy: {val_plumber_acc:.4f}")

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
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    parser.add_argument('--layers_to_mask', type=int, nargs='+', default=[10], help='Layers to mask for image prompting')
    parser.add_argument('--regularization_weight', type=float, default=0.05, help='Weight for regularization loss')
    
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
    parser.add_argument('--dataset_txt_prompt', action='store_true', help='Whether to use dataset level prompts or class level prompts')

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

    if args.use_saved_features and args.img_prompting:
        raise Exception("Cannot use saved features and learn image prompting")
    
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
python train_masked_task_distillation.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "cifar10" \
    --layers_to_mask 10 \
    --regularization_weight 0.05 \
    --train_on_testset \
    --num_classes 10 \
    --batch_size 128 \
    --seed 42 \
    --img_size 224 \
    --classifier_name "SimpleCNN" \
    --classifier_checkpoint_path "logs/cifar10/SimpleCNN/classifier/checkpoint_29.pth" \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/cifar10/cifar10_CLIP_ViT-B_32_text_embeddings.pth" \
    --n_promt_ctx 16 \
    --num_epochs 20 \
    --optimizer 'sgd' \
    --learning_rate 0.1 \
    --val_freq 1 \
    --prefix '' \
    --proj_clip \
    --projection_dim 512 \
    --teacher_temp 2 \
    --student_temp 1 \
    --weight_img_loss 1 \
    --weight_txt_loss 1 \
    --num_gpus 1 \
    --num_nodes 1


'''
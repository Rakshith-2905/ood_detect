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


from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier

from models.projector import ProjectionHead
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from models.plumber import PLUMBER

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
    projector_name += "_dataset_LP" if args.dataset_txt_prompt else ""
    projector_name += "_cls_LP" if args.cls_txt_prompts else ""

    att_name = ""
    if args.attributes:
        att_name = "".join([str(att) for att in args.attributes])
        att_name = f"att_{att_name}"
    else:
        att_name = "att_all"
    

    save_dir = os.path.join(args.save_dir, args.dataset_name, 'shift_detection', att_name, projector_name)
    
    save_dir_details = f"{args.prefix}_bs_{args.batch_size}_lr_{args.learning_rate}"
    save_dir_details += f"_teT_{args.teacher_temp}_sT_{args.student_temp}"
    save_dir_details += f"_imgweight_{args.weight_img_loss}_txtweight_{args.weight_txt_loss}_is_mlp_{args.is_mlp}"

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
      
def train_one_epoch(data_loader, plumber, 
                    class_prompts, text_encodings_raw, criterion, epoch):

    plumber.set_train_mode()

    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        plumber.zero_grad()


        image_encodings = plumber.encode_images(images_clip_batch)
        text_encodings_class = plumber.encode_text(class_prompts, text_encodings_raw)
            
        
        # Subselect the prompts and text encodings for the current batch
        batch_prompts = [class_prompts[label] for label in labels]
        text_encodings = text_encodings_class[labels]

        normalized_proj_embeddings = F.normalize(image_encodings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (batch_size, projection_dim)
        normalized_text_encodings_class = F.normalize(text_encodings_class, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        normalized_text_encodings_class = normalized_text_encodings_class.type_as(normalized_proj_embeddings)

        # T100 is the logits scale from CLIP
        logits_per_img = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
        logits_per_text = logits_per_img.t()
        
        # Logits for the class prompts (class level classification of images and class prompts)
        logits_per_img_class = 100*normalized_proj_embeddings @ normalized_text_encodings_class.t() # (batch_size, num_classes)
        
        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries
        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(image_encodings)).long().to(device) # (batch_size)

        loss_image = F.cross_entropy(logits_per_img, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss
        
        fabric.backward(loss)

        plumber.optimizer_step()

        probs_from_plumber = F.softmax(logits_per_img_class, dim=-1)

        batch_plumber_acc = compute_accuracy(probs_from_plumber, labels)

        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    return total_loss, total_plumber_acc, total_image_loss, total_text_loss

@torch.no_grad()
def validate(data_loader, plumber, 
                    class_prompts, text_encodings_raw, criterion, epoch):

    plumber.set_eval_mode()

    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Validating Epoch {epoch+1}"
    )
    
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        image_encodings = plumber.encode_images(images_clip_batch)
        text_encodings_class = plumber.encode_text(class_prompts, text_encodings_raw)
        
        # Subselect the prompts and text encodings for the current batch
        batch_prompts = [class_prompts[label] for label in labels]
        text_encodings = text_encodings_class[labels]

        normalized_proj_embeddings = F.normalize(image_encodings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (batch_size, projection_dim)
        normalized_text_encodings_class = F.normalize(text_encodings_class, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        normalized_text_encodings_class = normalized_text_encodings_class.type_as(normalized_proj_embeddings)

        # T100 is the logits scale from CLIP
        logits_per_img = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
        logits_per_text = logits_per_img.t()
        
        # Logits for the class prompts (class level classification of images and class prompts)
        logits_per_img_class = 100*normalized_proj_embeddings @ normalized_text_encodings_class.t() # (batch_size, num_classes)
        
        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries
        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(image_encodings)).long().to(device) # (batch_size)

        loss_image = F.cross_entropy(logits_per_img, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

        probs_from_plumber = F.softmax(logits_per_img_class, dim=-1)

        batch_plumber_acc = compute_accuracy(probs_from_plumber, labels)

        total_plumber_acc += batch_plumber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss

        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    return total_loss, total_plumber_acc, total_image_loss, total_text_loss

def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.eval()

    plumber = PLUMBER(args.clip_model_name, args.num_classes, img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, dataset_txt_prompt=args.dataset_txt_prompt, 
                      is_mlp=args.is_mlp, device=fabric.device)
    
    plumber = fabric.to_device(plumber)
    

    ########################### Load the dataset ############################

    train_transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                                    (0.229, 0.224, 0.225)),
                                ])
    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, class_names = get_dataset(args.dataset_name, train_transform, train_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, sample_by_attributes=args.attributes)
    
    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
    
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
        text_encodings = fabric.to_device(text_encodings)

        fabric.print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        fabric.print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")
        text_encodings = fabric.to_device(text_encodings)

    

    ########################### Create the optimizer ############################
    optimizers_dict = plumber.optimizer_init(args.optimizer, args.learning_rate)
    schedulers_dict = plumber.scheduler_init()

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
    
    start_epoch = 0

    state = {"clip_model": clip_model,
            "img_projector": plumber.img_projector, "text_projector": plumber.text_projector,
            "clip_prompted_txt_enc": plumber.clip_prompted_txt_enc, "clip_prompted_img_enc": plumber.clip_prompted_img_enc,
            "optimizer_img_proj": plumber.optimizer_img_proj,
            "optimizer_txt_proj": plumber.optimizer_txt_proj, "optimizer_txt_prompt": plumber.optimizer_txt_prompt, 
            "optimizer_img_prompt": plumber.optimizer_img_prompt, "epoch": start_epoch}

    if args.resume_checkpoint_path:
        fabric.load(args.resume_checkpoint_path, state)
        start_epoch = state["epoch"] + 1
        fabric.print(f"Resuming training from epoch {start_epoch}")
        # update the optimizers 

    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        
        train_loss, train_plumber_acc, train_loss_img, train_loss_txt = train_one_epoch(
                                                                            train_loader, plumber, class_prompts,
                                                                            text_encodings, criterion, epoch)
        if epoch % args.val_freq == 0:
            val_loss, val_plumber_acc, val_loss_img, val_loss_txt = validate(
                                                                            val_loader, plumber, class_prompts,
                                                                            text_encodings, criterion, epoch)
        
        plumber.scheduler_step()
            
        fabric.print(f"Epoch {epoch}/{args.num_epochs}| Train Loss: {train_loss:.4f}, Train PLUMBER Accuracy: {train_plumber_acc:.4f}, Val Loss: {val_loss:.4f}, Val PLUMBER Accuracy: {val_plumber_acc:.4f}")

        losses_dict = {"train_loss": train_loss, "train_loss_img": train_loss_img, "train_loss_txt": train_loss_txt,
                        "train_plumber_acc": train_plumber_acc,
                        "val_loss": val_loss, "val_loss_img": val_loss_img, "val_loss_txt": val_loss_txt, 
                        "val_plumber_acc": val_plumber_acc}

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
    parser.add_argument('--attributes', nargs='+', type=int, default=None, help='Attributes to use for training')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--train_on_testset', action='store_true', help='Whether to train on the test set or not')
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
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

    print = fabric.print

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
            
    seed_everything(args.seed)

    main(args)

'''
python train_on_data.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "NICOpp" \
    --attributes 5 \
    --num_classes 60 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --img_prompting --cls_txt_prompts  \
    --classifier_name 'resnet18' \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/NICOpp/NICOpp_CLIP_ViT-B_32_text_embeddings.pth" \
    --n_promt_ctx 16 \
    --num_epochs 10 \
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
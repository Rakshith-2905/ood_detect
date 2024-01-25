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

    save_dir = os.path.dirname(args.checkpoint_path)

    return os.path.join(save_dir, 'logs')

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
      
@torch.no_grad()
def validate(data_loader, plumber, 
                    class_prompts, text_encodings_raw, criterion, epoch):

    plumber.set_eval_mode()

    total_plumber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Validating Epoch {epoch+1}"
    )
    
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = fabric.to_device(images_batch)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        logits = plumber(images_clip_batch, class_prompts)

        probs_from_plumber = F.softmax(logits, dim=-1)

        batch_plumber_acc = compute_accuracy(probs_from_plumber, labels)

        total_plumber_acc += batch_plumber_acc

    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    return total_plumber_acc

def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.eval()

    plumber = PLUMBER(args.clip_model_name, args.num_classes, img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, dataset_txt_prompt=args.dataset_txt_prompt, 
                      is_mlp=args.is_mlp, device=fabric.device)
    
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            print(f"Checkpoint path {args.checkpoint_path} does not exist")
    else:
        plumber.load_checkpoint(args.checkpoint_path)
    
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
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, train_transform, 
                                                                                        data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                                        img_size=args.img_size, return_failure_set=True, 
                                                                                        sample_by_attributes=args.attributes, domain_name=args.domain_name)

    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
    
    fabric.print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")
    fabric.print(f"Number of failure examples: {len(failure_dataset)}")
    # Get the text encodings for the class names
    try:
        text_encodings= torch.load(args.prompt_path)
        if text_encodings.shape[0] != len(class_names):
            raise Exception("Text encodings shape does not match the number of classes")

        fabric.print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        fabric.print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    
    val_plumber_acc = validate(
                                val_loader, plumber, class_prompts,
                                text_encodings, None, 0)
    
    test_plumber_acc = validate(
                                test_loader, plumber, class_prompts,
                                text_encodings, None, 0)
    
    output_message = f"Val Acc: {val_plumber_acc:.4f}, Test Acc: {test_plumber_acc:.4f}"
    fabric.print(output_message)

    if fabric.is_global_zero:
        print(f"Writing results to {os.path.join(args.save_dir, f'{args.dataset_name}results.csv')}")
        # Save the results to a csv file in the save directory
        with open(os.path.join(args.save_dir, f'{args.dataset_name}results.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Val Acc', 'Test Acc'])
            writer.writerow([val_plumber_acc.item(), test_plumber_acc.item()])

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
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to load the step1 model from')

    parser.add_argument('--classifier_name',  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
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
    
    # with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
    #     for arg, value in vars(args).items():
    #         f.write(f"{arg}: {value}\n")

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
python test_on_data.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "NICOpp" \
    --attributes 1 2 3 \
    --num_classes 60 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --img_projection --txt_projection  \
    --checkpoint_path "logs/NICOpp/shift_detection/att_123/plumber_img_text_proj/_bs_256_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/best_projector_weights.pth" \
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
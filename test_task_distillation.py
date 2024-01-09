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
from torchvision import transforms

import torchvision.transforms as trn
import torchvision.datasets as dset
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

from data_utils.domainnet_data import DomainNetDataset, get_data_from_saved_files

from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18

from train_task_distillation import build_classifier, get_dataset, get_dataset_from_file, get_CLIP_text_encodings, PromptedCLIPTextEncoder


def get_save_dir(args):
    
    # If resume_checkpoint_path is provided, then use the save_dir from that checkpoint
    if args.resume_checkpoint_path:
        save_dir = os.path.dirname(args.resume_checkpoint_path)
        return save_dir

    save_dir = os.path.dirname(args.step1_checkpoint_path)

    save_dir = os.path.join(save_dir, 'logs')
    # save_dir += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return save_dir

class CIFAR10C(torch.utils.data.Dataset):
    def __init__(self, corruption='gaussian_blur', transform=None,clip_transform=None,level=0):
        numpy_path = f'data/CIFAR-10-C/{corruption}.npy'
        t = 10000
        self.transform = transform
        self.clip_transform = clip_transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:]
        self.data = self.data_[:t,:,:,:]
        self.targets_ = np.load('data/CIFAR-10-C/labels.npy')
        self.targets = self.targets_[:t]
        self.np_PIL = trn.Compose([trn.ToPILImage()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image_ = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image_)
            image_to_clip = self.clip_transform(self.np_PIL(image_))
        targets = self.targets[idx]
        return image, targets, image_to_clip

def get_dataset(data_name, train_transforms, test_transforms, clip_transform, data_dir='../data'):

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

    elif data_name == 'cifar10-c':
        not_needed = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform,selected_classes= None)
        class_names= not_needed.class_names

        train_dataset = CIFAR10C(corruption=args.domain_name, transform=test_transforms,clip_transform=clip_transform,level=args.severity)
        val_dataset = train_dataset
    else:
        raise ValueError(f"Invalid dataset name: {data_name}")
    
    return train_dataset, val_dataset, class_names

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
def validate(data_loader, clip_model, classifier, 
                    projector, projector_txt,
                    text_encodings_raw, class_prompts, clip_prompted, criterion, epoch):
    
    clip_model.eval()
    classifier.eval()
    if projector:
        projector.eval()
    if projector_txt:
        projector_txt.eval()

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

        clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted:
            text_encodings = clip_prompted(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = projector_txt(text_encodings_raw)
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
def validate_feat(data_loader, clip_model, classifier, 
                    projector, projector_txt,
                    text_encodings_raw, class_prompts, clip_prompted, criterion, epoch):
    clip_model.eval()
    classifier.eval()

    if projector: 
        projector.eval()
    if projector_txt:
        projector_txt.eval()

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

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)
        # Project the image embeddings
        if args.img_projection:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = classifier_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted:
            text_encodings = clip_prompted(class_prompts)

        # Project the text embeddings
        if args.txt_projection:
            text_encodings = projector_txt(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)
        # fabric.print(normalized_proj_embeddings.shape,normalized_text_encodings.shape, logits_projection.shape,classifier_logits.shape)
        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss

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
    
def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    projector = None
    if args.proj_clip:
        # This is PLUMBER
        projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
        fabric.print(f"Constructed img emb projection PLUMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
    else:
        # This is LIMBER
        projector = ProjectionHead(input_dim=classifier.feature_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
        fabric.print(f"Constructed img emb projection LIMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")

    step1_checkpoints = torch.load(args.step1_checkpoint_path)
    projector.load_state_dict(step1_checkpoints['projector'])
    projector = fabric.to_device(projector)

    text_projector = None
    if args.txt_projection:
        text_projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
        text_projector.load_state_dict(step1_checkpoints['text_projector'])
        text_projector = fabric.to_device(text_projector)
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
    clip_prompted = None
    if args.cls_txt_prompts:
        # Create the prompted CLIP model
        clip_prompted = PromptedCLIPTextEncoder(clip_model, n_ctx=args.n_promt_ctx, num_classes=len(class_names), device=fabric.device)
        clip_prompted = fabric.to_device(clip_prompted)

        clip_prompted.load_state_dict(step1_checkpoints['clip_prompted'])
        clip_prompted = fabric.to_device(clip_prompted)
        class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]

        fabric.print(f"Constructed CLIP Prompted Text Encoder with {len(class_names)} classes")


    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
        classifier = fabric.to_device(classifier)

    if args.use_saved_features:
        val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate_feat(val_loader, clip_model, classifier, projector, text_projector, 
                    text_encodings, class_prompts, clip_prompted, criterion, 0)
    else:    
        val_loss, val_base_acc, val_plumber_acc, val_loss_img, val_loss_txt = validate(val_loader, clip_model, classifier, projector, text_projector, 
                                       text_encodings, class_prompts, clip_prompted, criterion, 0)
        
        
    output_message = f"Dataset: {args.dataset_name} | {args.domain_name} - {args.severity}:\tVal Loss: {val_loss:.4f}, Val Base Model Accuracy: {val_base_acc:.4f}, Val PLUMBER Accuracy: {val_plumber_acc:.4f}"
    fabric.print(output_message)

    if fabric.is_global_zero:
        # Save the output to a text file
        output_file_path = os.path.join(args.save_dir, "validation_results.txt")
        with open(output_file_path, "a") as file:
            file.write(output_message + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--severity', type=int, default=3, help='Severity of the corruption')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--train_on_testset', action='store_true', help='Whether to train on the test set or not')
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    parser.add_argument('--img_projection', action='store_true', help='Whether to use task projection or not')
    parser.add_argument('--txt_projection', action='store_true', help='Whether to use text projection or not')
    parser.add_argument('--step1_checkpoint_path', type=str, help='Path to checkpoint to load the step 1 model from')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--n_promt_ctx', type=int, default=16, help='Number of learnable prompt token for each cls')
    parser.add_argument('--cls_txt_prompts', action='store_true', help='Whether to use learnable prompts or not')

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
    # fabric.print the arguments
    print(args)
    sys.stdout.flush()

    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    os.makedirs(args.save_dir, exist_ok=True)

    fabric = L.Fabric(accelerator="cuda",num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto")
   
    fabric.launch()

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
            
    seed_everything(args.seed)

    main(args)

'''
 python task_distill_eval.py \
        --data_dir './data/'  \
        --domain_name 'gaussian_blur'    \
        --dataset_name 'cifar10-c'    \
        --severity 3    \
        --train_on_testset    \
        --num_classes 10  \
        --batch_size 256  \
        --seed 42    \
        --img_projection \
        --cls_txt_prompts \
        --step1_checkpoint_path 'logs_2/cifar10/all/simple_cnn/plumber_img_proj_LP/_clsEpoch_29_bs_256_lr_0.1_teT_10.0_sT_1.0_imgweight_0.5_txtweight_8.0_is_mlp_False/step_1/projector_weights_final.pth' \
        --classifier_name 'SimpleCNN' \
        --classifier_checkpoint_path 'logs_2/cifar10/all/simple_cnn/classifier/model_epoch_29.pth' \
        --clip_model_name 'ViT-B/32' \
        --prompt_path 'data/cifar10/CiFAR10_CLIP_ViT-B_32_text_embeddings.pth' \
        --n_promt_ctx 16 \
        --num_epochs 10 \
        --optimizer 'sgd' \
        --learning_rate 0.1 \
        --val_freq 1 \
        --prefix '' \
        --proj_clip \
        --projection_dim 512 \
        --teacher_temp 0.5  \
        --student_temp 1 \
        --weight_img_loss 1 \
        --weight_txt_loss 1 \
        --num_gpus 1 \
        --num_nodes 1

'''
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

from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall, MulticlassCalibrationError
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

from data_utils.domainnet_data import DomainNetDataset, get_data_from_saved_files

from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18
from train_task_distillation import build_classifier, get_dataset_from_file, get_CLIP_text_encodings, get_dataset
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from models.custom_clip import MaskedTextTransformer, MaskedVisualTransformer

from ood_score import get_measures, print_measures, print_measures_with_std

def get_save_dir(args):
    
    save_dir = os.path.dirname(args.step1_checkpoint_path)
    save_dir = os.path.join(save_dir, 'logs')
    return save_dir

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
            masked_visual_encoder, text_encodings,
            criterion, epoch, metrics_computer):

    clip_model.eval()
    classifier.eval()

    def set_eval_mode(*models):
        for model in models:
            if model:
                model.eval()

    set_eval_mode(masked_visual_encoder)

    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    total_base_model_acc = 0
    total_plumber_acc = 0

    all_probs_from_classifier = []
    all_probs_from_proj = []
    all_labels = []

    all_logits_from_classifier = []
    all_logits_from_proj = []

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

        all_probs_from_classifier.append(probs_from_classifier)
        all_probs_from_proj.append(probs_from_proj)
        all_labels.append(labels)

        all_logits_from_classifier.append(classifier_logits.detach().cpu())
        all_logits_from_proj.append(logits_projection.detach().cpu())
        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_plumber_acc})
    
    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_image_loss = fabric.all_gather(total_image_loss).mean() / len(data_loader)
    total_text_loss = fabric.all_gather(total_text_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_plumber_acc = fabric.all_gather(total_plumber_acc).mean() / len(data_loader)

    all_probs_from_classifier = torch.cat(all_probs_from_classifier, dim=0)
    all_probs_from_proj = torch.cat(all_probs_from_proj, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_logits_from_classifier = torch.cat(all_logits_from_classifier, dim=0)
    all_logits_from_proj = torch.cat(all_logits_from_proj, dim=0)

    metrics = {}
    # metrics['base_model_acc'] = total_base_model_acc.item()
    # metrics['plumber_acc'] = total_plumber_acc.item()
    # # Expected Calibration Error
    # metrics['base_model_ece'] = metrics_computer['ece'](all_probs_from_classifier.detach().cpu(), all_labels.detach().cpu()).item()
    # metrics['plumber_ece'] = metrics_computer['ece'](all_probs_from_proj, all_labels).item()
    # metrics['base_model_precision'] = (metrics_computer['precision'](all_probs_from_classifier.detach().cpu(), all_labels.detach().cpu())).item()
    # metrics['plumber_precision'] = (metrics_computer['precision'](all_probs_from_proj.detach().cpu(), all_labels.detach().cpu())).item()
    # metrics['base_model_recall'] = (metrics_computer['recall'](all_probs_from_classifier.detach().cpu(), all_labels.detach().cpu())).item()
    # metrics['plumber_recall'] = (metrics_computer['recall'](all_probs_from_proj.detach().cpu(), all_labels.detach().cpu())).item()

    return total_loss, total_base_model_acc, total_plumber_acc, total_image_loss, total_text_loss, metrics, all_logits_from_classifier, all_logits_from_proj

def main(args):
    
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    # Load the checkpoint for the step 1 model
    checkpoint = torch.load(args.step1_checkpoint_path, map_location=fabric.device)
    visual_mask_layers = args.layers_to_mask
    masked_visual_encoder = MaskedVisualTransformer(clip_model, [10])
    masked_visual_encoder = fabric.to_device(masked_visual_encoder)
    masked_visual_encoder.load_state_dict(checkpoint['masked_visual_encoder'])

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
        train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                                data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                img_size=args.img_size, return_failure_set=True,
                                                                domain_name=args.domain_name, severity=args.severity)
        
        # Create the data loader and wrap them with Fabric
        _, _, OOD_test_dataset, _, _ = get_dataset('domainnet', train_transform, test_transform, 
                                                                data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                img_size=args.img_size, return_failure_set=True,
                                                                domain_name='sketch', severity=args.severity)
        fabric.print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    OOD_loader = torch.utils.data.DataLoader(OOD_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    val_loader, test_loader, OOD_loader = fabric.setup_dataloaders(val_loader, test_loader, OOD_loader)


    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")
    fabric.print(f"Number of OOD examples: {len(OOD_loader.dataset)}")
    # Get the text encodings for the class names
    try:
        text_encodings= torch.load(args.prompt_path)
        if text_encodings.shape[0] != len(class_names):
            raise Exception("Text encodings shape does not match the number of classes")

        fabric.print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        assert False, "{args.prompt_path} Prompt file not found."
        # text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        # fabric.print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
    fabric.print(f"Constructed CLIP Dataset specific Prompted Text Encoder with {len(class_names)} classes")

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    ece = MulticlassCalibrationError(num_classes=args.num_classes, n_bins=15, norm='l1')
    precision = MulticlassPrecision(num_classes=args.num_classes, average='macro')
    recall = MulticlassRecall(num_classes=args.num_classes, average='macro')

    metrics_computer = {
        "ece": ece,
        "precision": precision,
        "recall": recall
    }

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
        classifier = fabric.to_device(classifier)
    
    test_loss, test_base_acc, test_plumber_acc, test_loss_img, test_loss_txt, test_metrics, test_classifier_logits, test_plumber_logits = validate(
                                                                        test_loader, clip_model, classifier,
                                                                        masked_visual_encoder, text_encodings,
                                                                        criterion, 0, metrics_computer)
    
    _, _, _, _, _, _, ood_classifier_logits, ood_plumber_logits = validate(
                                                                        OOD_loader, clip_model, classifier,
                                                                        masked_visual_encoder, text_encodings,
                                                                        criterion, 0, metrics_computer)

    if args.score == 'MSP':
        id_classifier_score = torch.max(F.softmax(test_classifier_logits, dim=-1), dim=-1)[0].detach().cpu().numpy()
        ood_classifier_score = torch.max(F.softmax(ood_classifier_logits, dim=-1), dim=-1)[0].detach().cpu().numpy()

        id_plumber_score = torch.max(F.softmax(test_plumber_logits, dim=-1), dim=-1)[0].detach().cpu().numpy()
        ood_plumber_score = torch.max(F.softmax(ood_plumber_logits, dim=-1), dim=-1)[0].detach().cpu().numpy()

        print(id_classifier_score.shape, ood_classifier_score.shape)
        print(id_plumber_score.shape, ood_plumber_score.shape)

        classifier_auroc, classifier_aupr, classifier_fpr = get_measures(id_classifier_score, ood_classifier_score)
        plumber_auroc, plumber_aupr, plumber_fpr = get_measures(id_plumber_score, ood_plumber_score)
            
        print_measures(classifier_auroc, classifier_aupr, classifier_fpr, method_name='Classifier')
        print_measures(plumber_auroc, plumber_aupr, plumber_fpr, method_name='Plumber')

    assert False
    output_message = f"Validation Base Acc: {val_base_acc:.4f}, Validation PLUMBER Acc: {val_plumber_acc:.4f}, \n Test Base Acc: {test_base_acc:.4f}, Test PLUMBER Acc: {test_plumber_acc:.4f}"
    
    if args.dataset_name in ['cifar10-c', 'cifar100-c']:
        output_message = f"Corruption: {args.domain_name}, Severity: {args.severity}, " + output_message
    fabric.print(output_message)

    if fabric.is_global_zero:

        # convert the output to a dictionary, make the tensors to scalars
        output_dict = {'val_base_acc': val_base_acc.item(), 'val_plumber_acc': val_plumber_acc.item(),
                        'test_base_acc': test_base_acc.item(), 'test_plumber_acc': test_plumber_acc.item()}

        if args.dataset_name in ['cifar10-c', 'cifar100-c']:
            # Save the test_metrics to the same json file, add the corruption and severity to the dictionary
            test_metrics['corruption'] = args.domain_name
            test_metrics['severity'] = args.severity

            # Append to a json file
            save_path = os.path.join(args.save_dir, f"test_task_distillation_corruption.json")
            with open(save_path, 'a') as f:
                json.dump(test_metrics, f)
                f.write('\n')
            

            save_path = os.path.join(args.save_dir, f"test_task_distillation_corruption.txt")
            with open(save_path, 'a') as f:
                f.write(output_message)
                f.write('\n')
                return

        # Save the output to a csv file
        csv_file = os.path.join(args.save_dir, f"test_task_distillation.csv")
        with open(csv_file, 'w') as f:
            w = csv.writer(f)
            w.writerow(output_dict.keys())
            w.writerow(output_dict.values())
        
        # Save the test_metrics to a json file
        save_path = os.path.join(args.save_dir, f"test_task_distillation.json")
        with open(save_path, 'w') as f:
            json.dump(test_metrics, f)

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
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')


    parser.add_argument('--layers_to_mask', nargs='+', type=int, default=[10], help='List of layers to mask')
    parser.add_argument('--regularization_weight', type=float, default=0.05, help='Weight for the regularization loss')
    parser.add_argument('--step1_checkpoint_path', type=str, help='Path to checkpoint to load the step 1 model from')
    parser.add_argument('--score', type=str, choices=['MSP'], default='MSP', help='Score to use for OOD detection')

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
python test_masked_task_distillation.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "cifar10" \
    --layers_to_mask 10 \
    --regularization_weight 0.05 \
    --train_on_testset \
    --num_classes 10 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --classifier_name "SimpleCNN" \
    --classifier_checkpoint_path "logs/cifar10/SimpleCNN/classifier/checkpoint_29.pth" \
    --step1_checkpoint_path "logs/cifar10/SimpleCNN/masked_img_encoder_layer_10/_clsEpoch_29_bs_128_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_regweight_0.05/step_1/best_projector_weights.pth" \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/cifar10/cifar10_CLIP_ViT-B_32_text_embeddings.pth" \
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
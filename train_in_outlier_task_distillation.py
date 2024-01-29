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
from models.plumber import PLUMBER, LIMBER
from models.cluster import ClusterCreater

def get_save_dir(args):
    if args.resume_checkpoint_path:
        return os.path.dirname(args.resume_checkpoint_path)

    projector_name = "limber" if args.proj_clip else "limber"
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
    if args.dataset_name == "NICOpp":
        if args.attributes:
            att_name = "".join([str(att) for att in args.attributes])
            att_name = f"att_{att_name}"
        else:
            att_name = "att_all"
    
    if args.dataset_name == 'domainnet' and args.domain_name:
        att_name = f"{args.domain_name}"

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
      
def compute_clusters(data_loader, task_model, num_classes, samples_per_class, embedding_dim):

    cluster = ClusterCreater(10, samples_per_class, embedding_dim)

    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Computing Cluster"
    )
    for images_batch, labels, images_clip_batch in pbar:
        images_batch = images_batch[0]
        labels = labels[0]
        images_clip_batch = images_clip_batch[0]

        task_logits, image_embeddings = task_model(images_batch, return_features=True)
        predicted_class = torch.argmax(task_logits, dim=-1)

        # Get the indexes for correctly classified images and misclassified images
        correct_indexes = torch.where(predicted_class == labels)[0]

        # Using this compute pseudo labels for the images with 0 indicating incorrect and 1 indicating correct
        pseudo_labels = torch.zeros(len(labels))
        pseudo_labels[correct_indexes] = 1

        cluster.update_embeddings_batch(labels, image_embeddings)
    # virtual_embeddings = cluster.sample_embeddings_custom(1000, segment=4, class_specific=True)
    # head = task_model.fc3

    # class_logits = []
    # class_labels = []
    # for class_id, embeddings in enumerate(virtual_embeddings):
    #     class_logits.extend(head(embeddings.cuda()))
    #     class_labels.extend([class_id]*len(embeddings))

    # class_logits = torch.stack(class_logits)
    # class_labels = torch.tensor(class_labels).cuda()

    # # Compute the accuracy
    # acc = compute_accuracy(class_logits, class_labels)

    # print(acc)
    return cluster

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

def train_one_epoch(data_loader, task_virtual_cluster, clip_virtual_cluster, 
                    clip_model, limber, class_prompts, text_encodings_raw, 
                    criterion, epoch):

    limber.set_train_mode()

    total_loss = 0
    total_distill_loss = 0
    total_margin_loss = 0
    total_coral_loss = 0
    total_img_loss = 0
    total_txt_loss = 0

    total_base_model_acc = 0
    total_limber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Training Epoch {epoch+1}"
    )
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch_orig = images_batch[0]
        labels = labels[0]
        images_clip_batch = images_clip_batch[0]

        images_batch_aug = images_batch[1]

        images_batch_orig = fabric.to_device(images_batch_orig)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)

        limber.zero_grad()

        # Compute the original CLIP image embeddings
        orig_clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Compute projected LIMBER image and text embeddings
        classifier_logits, proj_image_embeddings = limber.encode_images(images_batch_orig, return_logits=True)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)

        text_encodings = limber.encode_text(class_prompts, text_encodings_raw)
        
        normalized_proj_embeddings = F.normalize(proj_image_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1).type_as(normalized_proj_embeddings)
        
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, class_size)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        # Compute distillation loss
        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        distill_loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss
        coral_loss = 0.2*coral_loss_compute(proj_image_embeddings, orig_clip_image_embeddings)
        
        if epoch > 5:
            try:

                task_prediction = torch.argmax(classifier_logits, dim=-1)
                # Get the indexes for correctly classified images and misclassified images by the task model
                task_correct_indexes = torch.where(task_prediction == labels)[0]

                # Using this compute pseudo labels for the images with 0 indicating incorrect and 1 indicating correct
                pseudo_labels = torch.zeros(len(labels))
                pseudo_labels[task_correct_indexes] = 1

                # # Update the virtual clusters with the pseudo labels from the task model and the proj image embeddings
                clip_virtual_cluster.update_embeddings_batch(labels, proj_image_embeddings)

                # Get the virtual embeddings
                virtual_outlier_embeddings = clip_virtual_cluster.sample_embeddings_custom(1000, segment=4, class_specific=False)
                # virtual_outlier_embeddings = clip_virtual_cluster.sample_embeddings(1000, select=100, class_specific=False)
                virtual_outlier_embeddings = fabric.to_device(virtual_outlier_embeddings)

                virtual_inlier_embeddings = task_virtual_cluster.sample_embeddings_custom(1000, segment=0, class_specific=False)
                # virtual_inlier_embeddings = task_virtual_cluster.sample_embeddings(1000, select=100, class_specific=False)
                virtual_inlier_embeddings = fabric.to_device(virtual_inlier_embeddings)

                virtual_proj_inlier_embeddings = limber.encode_features(virtual_inlier_embeddings)

                # Normalize the virtual embeddings
                normalized_virtual_outlier_embeddings = F.normalize(virtual_outlier_embeddings, dim=-1)
                virtual_clip_outlier_logits = 100*normalized_virtual_outlier_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
                virtual_clip_outlier_probs = F.softmax(virtual_clip_outlier_logits, dim=-1)

                # Normalize the virtual embeddings
                normalized_proj_virtual_inlier_embeddings = F.normalize(virtual_proj_inlier_embeddings, dim=-1)
                virtual_proj_inlier_logits = 100*normalized_proj_virtual_inlier_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
                virtual_proj_inlier_probs = F.softmax(virtual_proj_inlier_logits, dim=-1)

                margin_loss = 0.1*(torch.pow(F.relu(0.7-virtual_proj_inlier_probs), 2).mean() + 
                                torch.pow(F.relu(virtual_clip_outlier_probs-0.4), 2).mean())
                
                loss = distill_loss + margin_loss + coral_loss
            except:
                margin_loss = torch.tensor(0.0)
                loss = distill_loss + coral_loss
        else:
            margin_loss = torch.tensor(0.0)
            loss = distill_loss + coral_loss
        
        fabric.backward(loss)

        limber.optimizer_step()

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_limber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_limber_acc += batch_limber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss
        total_img_loss += loss_image.item()
        total_txt_loss += loss_text.item()

        total_distill_loss += distill_loss.item()
        total_margin_loss += margin_loss.item()
        total_coral_loss += coral_loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_img_loss = fabric.all_gather(total_img_loss).mean() / len(data_loader)
    total_txt_loss = fabric.all_gather(total_txt_loss).mean() / len(data_loader)
    total_distill_loss = fabric.all_gather(total_distill_loss).mean() / len(data_loader)
    total_margin_loss = fabric.all_gather(total_margin_loss).mean() / len(data_loader)
    total_coral_loss = fabric.all_gather(total_coral_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_limber_acc = fabric.all_gather(total_limber_acc).mean() / len(data_loader)

    performance_dict = {"total_loss": total_loss, "img_loss": total_img_loss,
                        "txt_loss": total_txt_loss, "distill_loss": total_distill_loss,
                        "margin_loss": total_margin_loss, "coral_loss": total_coral_loss,
                        "base_model_acc": total_base_model_acc, "limber_acc": total_limber_acc}
    
    return performance_dict

def validate(data_loader, task_virtual_cluster, clip_virtual_cluster, 
                    clip_model, limber, class_prompts, text_encodings_raw, 
                    criterion, epoch):

    limber.set_eval_mode()

    total_loss = 0
    total_distill_loss = 0
    total_margin_loss = 0
    total_coral_loss = 0
    total_img_loss = 0
    total_txt_loss = 0

    total_base_model_acc = 0
    total_limber_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Validating Epoch {epoch+1}"
    )
    
    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch_orig = images_batch
        labels = labels
        images_clip_batch = images_clip_batch

        images_batch_orig = fabric.to_device(images_batch_orig)
        images_clip_batch = fabric.to_device(images_clip_batch)
        labels = fabric.to_device(labels)
        # Compute the original CLIP image embeddings
        orig_clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Compute projected LIMBER image and text embeddings
        classifier_logits, proj_image_embeddings = limber.encode_images(images_batch_orig, return_logits=True)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)

        text_encodings = limber.encode_text(class_prompts, text_encodings_raw)
        
        normalized_proj_embeddings = F.normalize(proj_image_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1).type_as(normalized_proj_embeddings)
        
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, class_size)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        # Compute distillation loss
        loss_image = criterion(logits_projection, classifier_logits)
        loss_text = criterion(logits_projection.t(), classifier_logits.t())
        
        distill_loss = loss_image*args.weight_img_loss + loss_text*args.weight_txt_loss
        coral_loss = 0.2*coral_loss_compute(proj_image_embeddings, orig_clip_image_embeddings)
        
        if epoch > 5:
            try:

                task_prediction = torch.argmax(classifier_logits, dim=-1)
                # Get the indexes for correctly classified images and misclassified images by the task model
                task_correct_indexes = torch.where(task_prediction == labels)[0]

                # Using this compute pseudo labels for the images with 0 indicating incorrect and 1 indicating correct
                pseudo_labels = torch.zeros(len(labels))
                pseudo_labels[task_correct_indexes] = 1

                # # Update the virtual clusters with the pseudo labels from the task model and the proj image embeddings
                clip_virtual_cluster.update_embeddings_batch(labels, proj_image_embeddings)

                # Get the virtual embeddings
                virtual_outlier_embeddings = clip_virtual_cluster.sample_embeddings_custom(1000, segment=4, class_specific=False)
                # virtual_outlier_embeddings = clip_virtual_cluster.sample_embeddings(1000, select=100, class_specific=False)
                virtual_outlier_embeddings = fabric.to_device(virtual_outlier_embeddings)

                virtual_inlier_embeddings = task_virtual_cluster.sample_embeddings_custom(1000, segment=0, class_specific=False)
                # virtual_inlier_embeddings = task_virtual_cluster.sample_embeddings(1000, select=100, class_specific=False)
                virtual_inlier_embeddings = fabric.to_device(virtual_inlier_embeddings)

                virtual_proj_inlier_embeddings = limber.encode_features(virtual_inlier_embeddings)

                # Normalize the virtual embeddings
                normalized_virtual_outlier_embeddings = F.normalize(virtual_outlier_embeddings, dim=-1)
                virtual_clip_outlier_logits = 100*normalized_virtual_outlier_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
                virtual_clip_outlier_probs = F.softmax(virtual_clip_outlier_logits, dim=-1)

                # Normalize the virtual embeddings
                normalized_proj_virtual_inlier_embeddings = F.normalize(virtual_proj_inlier_embeddings, dim=-1)
                virtual_proj_inlier_logits = 100*normalized_proj_virtual_inlier_embeddings @ normalized_text_encodings.t() # (batch_size, batch_size)
                virtual_proj_inlier_probs = F.softmax(virtual_proj_inlier_logits, dim=-1)

                margin_loss = 0.1*(torch.pow(F.relu(0.7-virtual_proj_inlier_probs), 2).mean() + 
                                torch.pow(F.relu(virtual_clip_outlier_probs-0.4), 2).mean())
                
                loss = distill_loss + margin_loss + coral_loss
            except:
                margin_loss = torch.tensor(0.0)
                loss = distill_loss + coral_loss
        else:
            margin_loss = torch.tensor(0.0)
            loss = distill_loss + coral_loss
        
        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_limber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_limber_acc += batch_limber_acc

        batch_loss = loss.item() 
        total_loss += batch_loss
        total_img_loss += loss_image.item()
        total_txt_loss += loss_text.item()

        total_distill_loss += distill_loss.item()
        total_margin_loss += margin_loss.item()
        total_coral_loss += coral_loss.item()

    total_loss = fabric.all_gather(total_loss).mean() / len(data_loader)
    total_img_loss = fabric.all_gather(total_img_loss).mean() / len(data_loader)
    total_txt_loss = fabric.all_gather(total_txt_loss).mean() / len(data_loader)
    total_distill_loss = fabric.all_gather(total_distill_loss).mean() / len(data_loader)
    total_margin_loss = fabric.all_gather(total_margin_loss).mean() / len(data_loader)
    total_coral_loss = fabric.all_gather(total_coral_loss).mean() / len(data_loader)
    
    total_base_model_acc = fabric.all_gather(total_base_model_acc).mean() / len(data_loader)
    total_limber_acc = fabric.all_gather(total_limber_acc).mean() / len(data_loader)

    performance_dict = {"total_loss": total_loss, "img_loss": total_img_loss,
                        "txt_loss": total_txt_loss, "distill_loss": total_distill_loss,
                        "margin_loss": total_margin_loss, "coral_loss": total_coral_loss,
                        "base_model_acc": total_base_model_acc, "limber_acc": total_limber_acc}

    return performance_dict


def main(args):
    
    ########################### Create the model ############################
    clip_model, clip_transform = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()    

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)

    fabric.print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    if args.classifier_dim is None:
        args.classifier_dim = classifier.feature_dim

    limber = LIMBER(args.clip_model_name, args.num_classes, 
                     task_dims=args.classifier_dim, task_model=classifier,
                     img_projection=args.img_projection, txt_projection=args.txt_projection, 
                     img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, 
                     dataset_txt_prompt=args.dataset_txt_prompt, 
                     is_mlp=args.is_mlp, device=fabric.device)

    
    limber = fabric.to_device(limber)
    
    fabric.print(f"Built LIMBER with {args.clip_model_name} CLIP model")
    

    ########################### Load the dataset ############################

    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, domain_name=args.domain_name, 
                                                            return_failure_set=True)
    
    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
    fabric.print(f"Using {args.dataset_name} dataset")


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    # TODO: For any new datasets ensure this is handeled properly
    if args.train_on_testset:
        train_loader = val_loader
        fabric.print("Training on Val set")

    fabric.print(f"Number of training examples: {len(train_loader.dataset)}")
    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")
    fabric.print(f"Number of test examples: {len(test_loader.dataset)}")

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

    ########################### Compute the clusters ############################
    task_virtual_cluster = compute_clusters(train_loader, task_model=classifier, num_classes=2, 
                                            samples_per_class=100, embedding_dim=args.classifier_dim)

    clip_virtual_cluster = ClusterCreater(num_classes=10, samples_per_class=100, embedding_dim=args.projection_dim)

    ########################### Create the optimizer ############################
    optimizers_dict = limber.optimizer_init(args.optimizer, args.learning_rate)
    schedulers_dict = limber.scheduler_init()

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    if not args.use_saved_features:
        clip_model = fabric.to_device(clip_model)
    
    start_epoch = 0

    state = {"clip_model": clip_model,
            "img_projector": limber.img_projector, "text_projector": limber.text_projector,
            "clip_prompted_txt_enc": limber.clip_prompted_txt_enc, "clip_prompted_img_enc": limber.clip_prompted_img_enc,
            "optimizer_img_proj": limber.optimizer_img_proj,
            "optimizer_txt_proj": limber.optimizer_txt_proj, "optimizer_txt_prompt": limber.optimizer_txt_prompt, 
            "optimizer_img_prompt": limber.optimizer_img_prompt, "epoch": start_epoch}

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
        
        train_performance_dict = train_one_epoch(
                                            train_loader, task_virtual_cluster, clip_virtual_cluster, 
                                            clip_model, limber, class_prompts,
                                            text_encodings, criterion, epoch)
        

        if epoch % args.val_freq == 0:
            test_performance_dict = validate(
                                        test_loader, task_virtual_cluster, clip_virtual_cluster, 
                                        clip_model, limber, class_prompts,
                                        text_encodings, criterion, epoch)
        
        limber.scheduler_step()
        
        # Print the losses
        fabric.print(f"Epoch: {epoch+1}/{args.num_epochs} | Train Loss: {train_performance_dict['total_loss']:.4f} | Val Loss: {test_performance_dict['total_loss']:.4f} | Train Distill loss {train_performance_dict['distill_loss']:.4f} | Train Margin loss {train_performance_dict['margin_loss']:.4f} | Train Coral loss {train_performance_dict['coral_loss']:.4f} | Train Img loss {train_performance_dict['img_loss']:.4f} | Train Txt loss {train_performance_dict['txt_loss']:.4f} | Train Base Acc: {train_performance_dict['base_model_acc']:.4f} | Val Base Acc: {test_performance_dict['base_model_acc']:.4f}  | Train Limber Acc: {train_performance_dict['limber_acc']:.4f} | Val Limber Acc: {test_performance_dict['limber_acc']:.4f}")
        # Add train_ to all the keys
        train_performance_dict = {f"train_{key}": value for key, value in train_performance_dict.items()}
             
        # Add test_ to all the keys
        test_performance_dict = {f"test_{key}": value for key, value in test_performance_dict.items()}
        losses_dict = {**train_performance_dict, **test_performance_dict}


        fabric.log_dict(losses_dict, step=epoch)
        
        # Save best model based on validation loss
        if test_performance_dict["test_total_loss"] < best_val_loss:
            
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
    parser.add_argument('--classifier_dim', type=int, default=None, help='Dimension of the classifier output')

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

    fabric = L.Fabric(accelerator=args.device,num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto", loggers=[tb_logger, csv_logger])
   
    fabric.launch()

    print = fabric.print

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
    
            
    seed_everything(args.seed)

    main(args)

'''
python train_in_outlier_task_distillation.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "cifar10" \
    --img_projection --txt_projection \
    --train_on_testset \
    --num_classes 10 \
    --batch_size 128 \
    --seed 42 \
    --img_size 224 \
    --classifier_name "SimpleCNN" \
    --classifier_dim 84 \
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
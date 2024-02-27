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
import logging

from train_task_distillation import get_dataset, get_CLIP_text_encodings, build_classifier

from models.mapping import TaskMapping, MultiHeadedAttentionSimilarity, MultiHeadedAttention, print_layers, MeanAggregator, MaxAggregator
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, CutMix, MyAugMix, find_normalization_parameters, get_score, calc_gen_threshold, calc_accuracy_from_scores
from models.cluster import ClusterCreater
from sklearn.metrics import confusion_matrix
CLIP_LOGIT_SCALE = 100


class CIFAR100C(torch.utils.data.Dataset):
    def __init__(self, corruption='gaussian_blur', transform=None,clip_transform=None, level=0):
        numpy_path = f'data/CIFAR-100-C/{corruption}.npy'
        t = 10000 # We choose 10000 because, every numpy array has 50000 images, where the first 10000 images belong to severity 0 and so on. t is just an index for that
        self.transform = transform # Standard CIFAR100 test transform
        self.clip_transform = clip_transform
        self.data_ = np.load(numpy_path)[level*10000:(level+1)*10000,:,:,:] # Choosing 10000 images of a given severity
        self.data = self.data_[:t,:,:,:] # Actually redundant, I don't want to disturb the code structure
        self.targets_ = np.load('data/CIFAR-100-C/labels.npy')
        self.targets = self.targets_[:t] # We select the first 10000. The next 10000 is identical to the first 10000 and so on
        self.np_PIL = transforms.Compose([transforms.ToPILImage()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image_ = self.data[idx,:,:,:]
        if self.transform:
            image = self.transform(image_)
            image_to_clip = self.clip_transform(self.np_PIL(image_))
        targets = self.targets[idx]
        return image, targets, image_to_clip



def get_save_dir(args):
    
    if args.method == 'pim':
        if args.resume_checkpoint_path is not None and os.path.exists(args.resume_checkpoint_path):
            save_dir = os.path.dirname(args.resume_checkpoint_path)
        else:
            raise Exception("Checkpoint path not found")
    else:
        # use classifer checkpoint path
        if args.classifier_checkpoint_path is not None and os.path.exists(args.classifier_checkpoint_path):
            save_dir = os.path.dirname(args.classifier_checkpoint_path)
        else:
            raise Exception("Checkpoint path not found")

    save_dir = os.path.join(save_dir, 'failure_results')

    return f"{save_dir}"

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
def evaluate_pim(data_loader, class_attributes_embeddings, class_attribute_prompt_list,
                    clip_model, classifier, pim_model, aggregator): 
    
    # Set the model to eval mode
    pim_model.eval()
    aggregator.eval()
    classifier.eval()
    total_loss = 0
    total_task_model_acc = 0
    total_pim_acc = 0
    pbar = progbar_wrapper(
        data_loader, total=len(data_loader), desc=f"Eval"
    )
    
    labels_list, pim_logits_list, pim_probs_list = [], [], []
    task_model_logits_list, task_model_probs_list = [], []

    for i, (images_batch, labels, images_clip_batch) in enumerate(pbar):
        
        images_batch = images_batch.to(device)
        labels = labels.to(device)

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
        pim_logits = aggregator(pim_similarities_dict)

        loss = F.cross_entropy(pim_logits, labels)

        task_model_probs = F.softmax(task_model_logits, dim=-1)
        pim_probs = F.softmax(pim_logits, dim=-1)
        
        task_model_acc = compute_accuracy(task_model_probs, labels)
        pim_acc = compute_accuracy(pim_probs, labels)

        total_task_model_acc += task_model_acc
        total_pim_acc += pim_acc

        total_loss += loss.item()

        labels_list.append(labels)
        pim_logits_list.append(pim_logits)
        pim_probs_list.append(pim_probs)
        task_model_logits_list.append(task_model_logits)
        task_model_probs_list.append(task_model_probs)

    labels_list = torch.cat(labels_list, dim=0)
    pim_logits_list = torch.cat(pim_logits_list, dim=0)
    pim_probs_list = torch.cat(pim_probs_list, dim=0)
    task_model_logits_list = torch.cat(task_model_logits_list, dim=0)
    task_model_probs_list = torch.cat(task_model_probs_list, dim=0)
    

    print(labels_list.shape, pim_logits_list.shape, pim_probs_list.shape, task_model_logits_list.shape, task_model_probs_list.shape)
    pim_acc = compute_accuracy(pim_probs_list, labels_list)
    task_model_acc = compute_accuracy(task_model_probs_list, labels_list)
    print(f'PIM Accuracy on {args.dataset_name} = {pim_acc} and Task Model Accuracy = {task_model_acc}')
    return pim_acc, task_model_acc, labels_list, pim_logits_list, pim_probs_list, task_model_logits_list, task_model_probs_list

def main(args):

    log_path = f'{args.save_dir}/gen'
    
    
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

    if args.eval_dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    elif args.eval_dataset == 'cifar100c':
        transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        testset = CIFAR100C(corruption=args.cifar100c_corruption, transform=transform_test,clip_transform=clip_transform, level=args.severity)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Number of validation examples: {len(val_loader.dataset)}")
    print(f"Number of test examples: {len(test_loader.dataset)}")


    print(f"Loaded checkpoint from {args.classifier_checkpoint_path}")
    
    if args.method == 'baseline':
        classifier.to(device)
        classifier.eval()
        
        # Evaluating task model
        print('Evaluating on Validation Data')
        val_acc, val_labels_list, val_logits_list, val_probs_list = evaluate_classifier(val_loader, classifier, device=device)
        val_scores = get_score(args.score, val_logits_list)
        threshold = calc_gen_threshold(val_scores, val_logits_list, val_labels_list, name='classifier')

        # Just for verification
        estimated_val_acc, val_estimated_success_failure_idx = calc_accuracy_from_scores(val_scores, threshold)

        # Repeating this for test data
        print('Evaluating on Test Data')
        test_acc, test_labels_list, test_logits_list, test_probs_list = evaluate_classifier(test_loader, classifier, device=device)
        test_scores = get_score(args.score, test_logits_list)
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
        print(f'Success Recall = {cm_val[1,1]/(cm_val[1,0]+cm_val[1,1])}')

        print('Test Data')
        print(cm_test)
        print(f'Gen Gap = {torch.abs(test_acc-estimated_test_acc)}')
        print(f'Failure Recall = {cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])}')
        print(f'Success Recall = {cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])}')

        if args.eval_dataset == 'cifar100c':
            # convert confusion matrix to list
            cm_val_list = cm_val.tolist()
            cm_test_list = cm_test.tolist()
            # Convert the results to a dictionary
            results = {
                "cifar100c_corruption": args.cifar100c_corruption,
                "severity": args.severity,
                "true_val_acc": val_acc,
                "estimated_val_acc": estimated_val_acc.item(),
                "true_test_acc": test_acc,
                "estimated_test_acc": estimated_test_acc.item(),
                "val_cm": cm_val_list, # Confusion matrix for validation data as a list
                "test_cm": cm_test_list,
                "val_failure_recall": cm_val[0,0]/(cm_val[0,0]+cm_val[0,1]),
                "val_success_recall": cm_val[1,1]/(cm_val[1,0]+cm_val[1,1]),
                "test_failure_recall": cm_test[0,0]/(cm_test[0,0]+cm_test[0,1]),
                "test_success_recall": cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])
            }

            # Save the results to a file
            results_file = f'{args.save_dir}/{args.score}_cifar100c_results.json'
            with open(results_file, 'a') as f:
                json.dump(results, f)
                f.write('\n')

    elif args.method == 'pim':
        class_attributes_embeddings_prompts = torch.load(args.attributes_embeddings_path)
        class_attribute_prompts = class_attributes_embeddings_prompts["class_attribute_prompts"]
        class_attributes_embeddings = class_attributes_embeddings_prompts["class_attributes_embeddings"]

        assert len(class_attribute_prompts) == args.num_classes, "Number of classes does not match the number of class attributes"

        num_attributes_per_cls = [len(attributes) for attributes in class_attribute_prompts]
        
        
        if args.attribute_aggregation == "mha":
            aggregator = MultiHeadedAttentionSimilarity(args.num_classes, num_attributes_per_cls=num_attributes_per_cls, num_heads=1, out_dim=1)
        elif args.attribute_aggregation == "mean":
            aggregator = MeanAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)

        elif args.attribute_aggregation == "max":
            aggregator = MaxAggregator(num_classes=args.num_classes, num_attributes_per_cls=num_attributes_per_cls)

        else:
            raise Exception("Invalid attribute aggregation method")

        if args.resume_checkpoint_path:
            state = torch.load(args.resume_checkpoint_path)
            print(state.keys())
            classifier.load_state_dict(state["classifier"])
            pim_model.load_state_dict(state["pim_model"])
            aggregator.load_state_dict(state[f"aggregator"])

            print(f"Loaded checkpoint from {args.resume_checkpoint_path}")


        print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")
        print(f"Built {args.classifier_name} mapper")
        print(f"Built MultiHeadedAttention with {args.num_classes} classes and {num_attributes_per_cls} attributes per class")

        clip_model.to(device)
        classifier.to(device)
        pim_model.to(device)
        aggregator.to(device)

        # Evaluating task model
        print('Evaluating on Validation Data')
        # TODO: Update this to get task model logits and probs as well
        outs = evaluate_pim(val_loader, class_attributes_embeddings, class_attribute_prompts,
                            clip_model, classifier, pim_model, aggregator)
        
        val_pim_acc, val_task_model_acc, val_labels_list, val_pim_logits_list, val_pim_probs_list, val_task_logits_list, val_task_probs_list = outs
        val_scores = get_score(args.score, val_task_logits_list, val_pim_logits_list)

        threshold = calc_gen_threshold(val_scores, val_task_logits_list, val_labels_list, name='pim')

        estimated_val_acc, val_estimated_success_failure_idx = calc_accuracy_from_scores(val_scores, threshold)

        # Repeating this for test data
        print('Evaluating on Test Data')
        outs = evaluate_pim(test_loader, class_attributes_embeddings, class_attribute_prompts,
                            clip_model, classifier, pim_model, aggregator)
        
        test_pim_acc, test_task_model_acc, test_labels_list, test_pim_logits_list, test_pim_probs_list, test_task_logits_list, test_task_probs_list = outs
        test_scores = get_score(args.score, test_task_logits_list, test_pim_logits_list)
        estimated_test_acc, test_estimated_success_failure_idx = calc_accuracy_from_scores(test_scores, threshold)

        print(f'Score = {args.score}')
        print(f'True Validation Accuracy = {val_task_model_acc}, Estimated Validation Accuracy = {estimated_val_acc}, True Test Accuracy = {test_task_model_acc}, Estimated Test Accuracy = {estimated_test_acc}')
        val_true_success_failure_idx = torch.argmax(val_task_logits_list, 1) == val_labels_list
        test_true_success_failure_idx = torch.argmax(test_task_logits_list, 1) == test_labels_list

        print('Confusion Matrices')
        cm_val = confusion_matrix(val_true_success_failure_idx.cpu().numpy(), val_estimated_success_failure_idx.cpu().numpy())
        cm_test = confusion_matrix(test_true_success_failure_idx.cpu().numpy(), test_estimated_success_failure_idx.cpu().numpy())

        print('Validation Data')
        print(cm_val)
        print(f'Gen Gap = {torch.abs(val_task_model_acc-estimated_val_acc)}')
        print(f'Failure Recall = {cm_val[0,0]/(cm_val[0,0]+cm_val[0,1])}')
        print(f'Success Recall = {cm_val[1,1]/(cm_val[1,0]+cm_val[1,1])}')

        print('Test Data')
        print(cm_test)
        print(f'Gen Gap = {torch.abs(test_task_model_acc-estimated_test_acc)}')
        print(f'Failure Recall = {cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])}')
        print(f'Success Recall = {cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])}')

        if args.eval_dataset == 'cifar100c':
            # convert confusion matrix to list
            cm_val_list = cm_val.tolist()
            cm_test_list = cm_test.tolist()
            # Convert the results to a dictionary
            results = {
                "cifar100c_corruption": args.cifar100c_corruption,
                "severity": args.severity,
                "true_val_acc": val_task_model_acc,
                "estimated_val_acc": estimated_val_acc.item(),
                "true_test_acc": test_task_model_acc,
                "estimated_test_acc": estimated_test_acc.item(),
                "val_cm": cm_val_list, # Confusion matrix for validation data as a list
                "test_cm": cm_test_list,
                "val_failure_recall": cm_val[0,0]/(cm_val[0,0]+cm_val[0,1]),
                "val_success_recall": cm_val[1,1]/(cm_val[1,0]+cm_val[1,1]),
                "test_failure_recall": cm_test[0,0]/(cm_test[0,0]+cm_test[0,1]),
                "test_success_recall": cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])
            }

            # Save it as a CSV file
            results_file = f'{args.save_dir}/cifar100c_results.json'
            with open(results_file, 'a') as f:
                json.dump(results, f)
                f.write('\n')
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
    parser.add_argument('--eval_dataset', type=str, default='cifar100', help='Evaluation dataset')
    parser.add_argument('--filename', type=str, default='cifar100c.log', help='Filename')
    parser.add_argument('--cifar100c_corruption', default="gaussian_blur", type=str, help='Corruption type')
    parser.add_argument('--severity', default=5, type=int, help='Severity of corruption')
    
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--task_layer_name', type=str, default='model.layer2', help='Name of the layer to use for the task model')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for cutmix')
    parser.add_argument('--augmix_severity', type=int, default=3, help='Severity of the augmix')
    parser.add_argument('--augmix_alpha', type=float, default=1.0, help='Alpha value for the beta distribution for augmix')
    parser.add_argument('--augmix_prob', type=float, default=0.2, help='Probability of using augmix')
    parser.add_argument('--cutmix_prob', type=float, default=0.2, help='Probability of using cutmix')

    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs before using cutmix')
    parser.add_argument('--discrepancy_weight', type=float, default=1.0, help='Weight to multiply the loss by for samples where the task model is correct and the pim model is incorrect')

    parser.add_argument('--attributes_path', type=str, help='Path to the attributes file')
    parser.add_argument('--attributes_embeddings_path', type=str, help='Path to the attributes embeddings file')

    parser.add_argument('--attribute_aggregation', default='mha', choices=['mha', 'mean', 'max'], help='Type of aggregation of the attribute scores')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--classifier_dim', type=int, default=None, help='Dimension of the classifier output')

    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--aggregator_learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, choices=['MultiStepLR', 'cosine'], default='cosine', help='Type of learning rate scheduler to use')
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
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\nResults will be saved to {args.save_dir}")
    
    corruption_list = ['brightness', 'defocus_blur', 'fog', 'gaussian_blur', 'glass_blur', 'jpeg_compression', 'motion_blur', 'saturate','snow','speckle_noise', 'contrast', 'elastic_transform', 'frost', 'gaussian_noise', 'impulse_noise', 'pixelate','shot_noise', 'spatter','zoom_blur']
    severity = [4]
    if args.eval_dataset == 'cifar100c':
        for c in corruption_list:
            for s in severity:
                args.cifar100c_corruption = c
                args.severity = s
                print(f'Corruption = {args.cifar100c_corruption}, Severity = {args.severity}')
                seed_everything(args.seed)
                main(args)
    else:
        seed_everything(args.seed)
        main(args)

'''
Example usage:

python failure_detection_eval.py \
--data_dir './data' \
--dataset_name cifar100 \
--num_classes 100 \
--batch_size 512 \
--img_size 32 \
--seed 42 \
--task_layer_name model.layer3 \
--cutmix_alpha 1.0 \
--warmup_epochs 10 \
--discrepancy_weight 1.0 \
--attributes_path clip-dissect/cifar100_core_concepts.json \
--attributes_embeddings_path data/cifar100/cifar100_attributes_CLIP_ViT-B_32_text_embeddings.pth \
--classifier_name resnet18 \
--classifier_checkpoint_path logs/cifar100/resnet18/classifier/checkpoint_199.pth \
--use_imagenet_pretrained \
--attribute_aggregation mean \
--clip_model_name ViT-B/32 \
--prompt_path data/cifar100/cifar100_CLIP_ViT-B_32_text_embeddings.pth \
--num_epochs 200 \
--optimizer adamw \
--learning_rate 1e-3 \
--aggregator_learning_rate 1e-3 \
--scheduler MultiStepLR \
--val_freq 1 \
--save_dir ./logs \
--prefix '' \
--vlm_dim 512 \
--num_gpus 1 \
--num_nodes 1 \
--augmix_prob 0.2 \
--cutmix_prob 0.2 \
--resume_checkpoint_path logs/cifar100/mapper/_agg_mean_bs_512_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_layer_model.layer3/pim_weights_best.pth \
--method baseline \
--score msp \
--eval_dataset cifar100c \
--filename cifar100c.log


'''
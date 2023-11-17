import os
import sys
import time
import torch
import torch.nn.functional as F

import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torch.utils.data.dataset import Subset

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
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix

from domainnet_data import DomainNetDataset, get_data_from_saved_files

from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from YFCC_feature_extract import ImageTextDataset
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from train_projection_distill_cont import build_classifier


def class_level_entropies(entropy_, label_list, num_classes):
    class_entropies = {}
    for class_idx in range(num_classes):
        class_entropies[class_idx]=0
    
    for class_idx in range(num_classes):
        class_mask = (label_list == class_idx)
        class_entropies[class_idx] += np.sum(entropy_[class_mask])
    class_entropy_list=[]
    for class_idx in range(num_classes):
        class_lengths= len(label_list[label_list==class_idx])
        class_entropies[class_idx] /= class_lengths
        class_entropy_list.append(class_entropies[class_idx])

    return class_entropy_list

def entropy(prob):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)

@torch.no_grad()
def get_entropy_confusion(val_loader,classifier,clip_model,clip_text_encodings,projector,PROJ_CLIP,Teacher_Temp, device):

    all_labels = []
    classifier_prob_list, proj_prob_list, CLIP_prob_list = [], [], []
    clip_text_encodings=clip_text_encodings.to(device)


    for i,(images_batch, labels, images_clip_batch) in enumerate(val_loader):
        print(f"batch: {i}")
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)    
        labels = labels.to(device)
        all_labels.append(labels.cpu())
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)
        
        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        if PROJ_CLIP: # this is PLUMBER
            proj_embeddings = projector(clip_image_embeddings) # (batch_size, projection_dim)
        else: # this is LIMBER
            proj_embeddings = projector(classifier_embeddings) # (batch_size, projection_dim)

        normalized_clip_embeddings = F.normalize(clip_image_embeddings, dim=-1)
        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(clip_text_encodings, dim=-1)
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)

        # T100 is the logits scale from CLIP
        projection_logits = 100 * normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)
        CLIP_logits = 100 * normalized_clip_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        classifier_logits = classifier_logits / Teacher_Temp
        
        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(projection_logits, dim=-1)
        probs_from_CLIP = F.softmax(CLIP_logits, dim=-1)

        classifier_prob_list.append(probs_from_classifier)
        proj_prob_list.append(probs_from_proj)
        CLIP_prob_list.append(probs_from_CLIP)
        
    classifier_prob_list = torch.cat(classifier_prob_list, dim=0)
    proj_prob_list = torch.cat(proj_prob_list, dim=0)
    CLIP_prob_list = torch.cat(CLIP_prob_list, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute Confusion Matrix
    _, predicted_classifier = torch.max(classifier_prob_list, 1)
    _, predicted_proj = torch.max(proj_prob_list, 1)
    _, predicted_CLIP = torch.max(CLIP_prob_list, 1)

    confusion_matrix_classifier = confusion_matrix(all_labels.cpu().numpy().ravel(), predicted_classifier.cpu().numpy().ravel())
    confusion_matrix_proj = confusion_matrix(predicted_classifier.cpu().numpy().ravel(), predicted_proj.cpu().numpy().ravel())
    confusion_matrix_CLIP = confusion_matrix(all_labels.cpu().numpy().ravel(), predicted_CLIP.cpu().numpy().ravel())

    return classifier_prob_list, proj_prob_list, CLIP_prob_list, all_labels, confusion_matrix_classifier, confusion_matrix_proj, confusion_matrix_CLIP

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_dir=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
    else:
            print('Confusion matrix, without normalization')

    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_dir:
        plt.savefig(save_dir)
    plt.close()

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
        # train_dataset = dset.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transforms)
        # val_dataset = dset.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transforms)
        # class_names = train_dataset.classes

        # # Selecting classes 0, 1, and 2
        # train_indices = [i for i, (_, y) in enumerate(train_dataset) if y in [0, 1, 2]]
        # test_indices = [i for i, (_, y) in enumerate(val_dataset) if y in [0, 1, 2]]

        # train_dataset = Subset(train_dataset, train_indices)
        # val_dataset = Subset(val_dataset, test_indices)
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform)

        class_names = ['airplane', 'automobile', 'bird']

    return train_dataset, val_dataset, class_names

def get_save_dir(args):
    
    # If resume_checkpoint_path is provided, then use the save_dir from that checkpoint
    if args.resume_checkpoint_path:
        save_dir = os.path.dirname(args.resume_checkpoint_path)
        return save_dir

    save_dir = os.path.join(args.save_dir, args.classifier_name)

    save_dir += f"{args.prefix}"
    save_dir += f"_bs{args.batch_size}"
    save_dir += f"_teT_{args.teacher_temp}_sT_{args.student_temp}"
    save_dir += f"_imgweight_{args.weight_img_loss}_txtweight_{args.weight_txt_loss}"
    save_dir += f"_is_mlp_{args.is_mlp}"
    
    # save_dir += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return save_dir

@torch.no_grad()
def get_CLIP_text_encodings(clip_model, texts, save_path=None, device='cuda'):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_encodings = clip_model.encode_text(text_tokens).float()
    # text_encoding_save_path = os.path.join(os.getcwd(), "imagenet_classes_text_encodings.pt")
    torch.save(text_encodings,save_path )
    return text_encodings

def plot_entropy(args, save_dir, classifier_entropy, proj_entropy, CLIP_entropy, 
                 entropy_classifier, entropy_proj, entropy_CLIP):

    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs=np.asarray(axs).reshape(1,3)
        
    axs[0,0].stem(range(args.num_classes), classifier_entropy[args.domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
    axs[0,1].stem(range(args.num_classes), proj_entropy[args.domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
    axs[0,2].stem(range(args.num_classes), CLIP_entropy[args.domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/classwise_entropy_epo_{args.num_epochs}.png")
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    
    axs[0].hist( entropy_classifier[args.domain_name] )
    axs[1].hist( entropy_proj[args.domain_name] )
    axs[2].hist( entropy_CLIP[args.domain_name])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/entropy_hist_epo_{args.num_epochs}.png")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.to(device)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    if args.proj_clip:
        projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
    else:
        projector = ProjectionHead(input_dim=classifier.feature_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
    
    save_dir = get_save_dir(args)

    # Load the checkpoint
    checkpoint_path = os.path.join(save_dir, f'projector_weights_{args.num_epochs}.pth')
    projector.load_state_dict(torch.load(checkpoint_path)['projector'])
    print(f"Loaded checkpoint from {checkpoint_path}")

    projector = projector.to(device)
    projector.eval()
    classifier = classifier.to(device)
    classifier.eval()

    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform,
                                                                data_dir=args.data_dir, clip_transform=clip_transform)
  
                
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)


    text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path, device=device)
    print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    classifier_prob_list, proj_prob_list, CLIP_prob_list, label_list, confusion_matrix_classifier, confusion_matrix_proj, confusion_matrix_CLIP =  get_entropy_confusion(val_loader, classifier, clip_model, text_encodings, projector,
                                                                                   args.proj_clip, args.teacher_temp, device)
    print(f"CLIP_prob_list: {CLIP_prob_list.shape}", f"proj_prob_list: {proj_prob_list.shape}, classifier_prob_list: {classifier_prob_list.shape}")
        
    plot_confusion_matrix(confusion_matrix_classifier, class_names, 
                          normalize=False, title='Confusion matrix for classifier', save_dir=f"{save_dir}/confusion_matrix_classifier_{args.num_epochs}.png")
    plot_confusion_matrix(confusion_matrix_proj, class_names,
                            normalize=False, title='Confusion matrix for proj', save_dir=f"{save_dir}/confusion_matrix_proj_{args.num_epochs}.png")
    plot_confusion_matrix(confusion_matrix_CLIP, class_names,
                            normalize=False, title='Confusion matrix for CLIP', save_dir=f"{save_dir}/confusion_matrix_CLIP_{args.num_epochs}.png")

    classifier_entropy={}
    proj_entropy={}
    CLIP_entropy={}

    entropy_classifier={}
    entropy_proj={}
    entropy_CLIP={}

    entropy_classifier[args.domain_name] = entropy(classifier_prob_list.cpu().data.numpy())
    entropy_proj[args.domain_name] = entropy(proj_prob_list.cpu().data.numpy())
    entropy_CLIP[args.domain_name] = entropy(CLIP_prob_list.cpu().data.numpy())


    label_list=label_list.cpu().numpy()

    classifier_entropy[args.domain_name] = class_level_entropies(entropy_classifier[args.domain_name], label_list,args.num_classes)
    proj_entropy[args.domain_name] = class_level_entropies(entropy_proj[args.domain_name], label_list,args.num_classes)
    CLIP_entropy[args.domain_name] = class_level_entropies(entropy_CLIP[args.domain_name], label_list,args.num_classes)


    with open(os.path.join(save_dir,'entropies.pkl'), "wb") as f:   
        data={}
        data["classifier_prob_list"]=classifier_prob_list
        data["proj_prob_list"]=proj_prob_list
        data["CLIP_prob_list"]=CLIP_prob_list
        data["label_list"]=label_list

        data["entropy_classifier"]=entropy_classifier[args.domain_name]
        data["entropy_proj"]=entropy_proj[args.domain_name]
        data["entropy_CLIP"]=entropy_CLIP[args.domain_name]
        #classifier_prob_list, proj_prob_list, CLIP_prob_list, label_list
    
        data["classifier_entropy"] = classifier_entropy[args.domain_name]
        data["proj_entropy"] = proj_entropy[args.domain_name]
        data["CLIP_entropy"] = CLIP_entropy[args.domain_name]
        pickle.dump(data, f)

    plot_entropy(args, save_dir, classifier_entropy, proj_entropy, 
                 CLIP_entropy, entropy_classifier, entropy_proj, entropy_CLIP)

#Checklist
# Did you change the dataset name?
# Did you change the domain name?
# Did you change the scale?
# Did you change the classifier name?
# Did you change the projector weights path?
# did you change num_classes?

####################


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
    
    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the results')
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

    args = parser.parse_args()
    main(args)
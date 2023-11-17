import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
import torchvision.datasets as dset

import sys
sys.path.insert(0, '../')
from domainnet_data import DomainNetDataset, get_domainnet_loaders
from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead

from simple_classifier import SimpleCNN, CIFAR10TwoTransforms
from train_projection_distill_cont import build_classifier

import clip
from collections import OrderedDict

torch.manual_seed(0)

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "data/broden1_224/images/"}

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def get_dataset(data_name, train_transforms, test_transforms, clip_transform, data_dir='../data', domain_name=None):

    if data_name == 'imagenet':
        train_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_train_examples', transform=train_transforms)
        val_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_val_examples', transform=test_transforms)
        class_names = train_dataset.classes

    elif data_name == 'domainnet':
        train_dataset = DomainNetDataset(root_dir=data_dir, domain=domain_name, \
                                        split='train', transform=train_transforms, transform2=clip_transform)
        val_dataset = DomainNetDataset(root_dir=data_dir, domain=domain_name, \
                                        split='test', transform=test_transforms, transform2=clip_transform)
        class_names = train_dataset.class_names

    elif data_name == 'cifar10':
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform, selected_classes= [0,1,2])
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform, selected_classes= [0,1,2])

        class_names = ['airplane', 'automobile', 'bird']
    elif data_name =="cifar10_full":
        
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform,selected_classes= None)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform,selected_classes= None)
        class_names= train_dataset.class_names


    return train_dataset, val_dataset, class_names

@torch.no_grad()
def get_CLIP_text_encodings(clip_model, texts, save_path=None):
    device = clip_model.device
    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_tokens = device(text_tokens)
        text_encodings = clip_model.encode_text(text_tokens).float()
    # text_encoding_save_path = os.path.join(os.getcwd(), "imagenet_classes_text_encodings.pt")
    torch.save(text_encodings,save_path )
    return text_encodings

def get_target_model( target_name, device, domain=None, projector_checkpoint_path=None):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == "plumber":

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        convert_models_to_fp32(clip_model)
        
        projector = ProjectionHead(input_dim=512, output_dim=512, is_mlp=False).to(device)
        projector.load_state_dict(torch.load(projector_checkpoint_path)["projector"])
        
        target_model = nn.Sequential(OrderedDict([
            ('feature_extractor', clip_model.visual),
            ('projector', projector)
        ])).to(device)
        
    elif target_name == "CLIP_RN50":
        clip_model, _ = clip.load("RN50", device=device)
        target_model = clip_model.visual
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    target_model.eval()
    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None, domain=None, data_dir='./data'):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    
    elif dataset_name == 'custom_domainnet':

        if domain == "spc":
            domains_interest = ['clipart', 'painting', 'sketch']
            combined_data = []
            for domain in domains_interest:
                data = DomainNetDataset(root_dir='data/domainnet_v1.0', domain=domain, split='test', transform=preprocess)
                combined_data.append(data)
            data = ConcatDataset(combined_data)
        else:
            data = DomainNetDataset(root_dir='data/domainnet_v1.0', domain=domain, split='probe', transform=preprocess)

        if len(data) > 20000:
            # Randomly subsample 20000 images from the dataset
            data = torch.utils.data.Subset(data, torch.randperm(len(data))[:20000])
    
    elif dataset_name == 'custom_cifar10':
        data = CIFAR10TwoTransforms(root=f'./data/cifar10', train=False, transform1=preprocess)

        class_names = ['airplane', 'automobile', 'bird']
    
    elif dataset_name in ['cifar10', 'custom_imagenet', 'custom_domainnet', 'custom_cifar10']:
    
        # data_name = dataset_name.split('_')[1]

        train_dataset, val_dataset, class_names =  get_dataset(dataset_name, train_transforms=preprocess, test_transforms=preprocess, 
                                                               clip_transform=None,data_dir=data_dir, domain_name=domain)
        data = val_dataset

    return data

def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass
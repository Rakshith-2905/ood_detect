import os
import torch
import pandas as pd
from torchvision import datasets, transforms, models
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
import sys
sys.path.insert(0, '../')
from domainnet_data import DomainNetDataset, get_domainnet_loaders
from models.resnet import CustomClassifier, CustomResNet
from models.projector import ProjectionHead

from simple_classifier import SimpleCNN, CIFAR10TwoTransforms

import clip
from collections import OrderedDict

torch.manual_seed(0)

DATASET_ROOTS = {"imagenet_val": "YOUR_PATH/ImageNet_val/",
                "broden": "data/broden1_224/images/"}


def build_classifier(classifier_name, num_classes, pretrained=False, checkpoint_path=None):

    if classifier_name in ['vit_b_16', 'swin_b']:
        classifier = CustomClassifier(classifier_name, use_pretrained=pretrained)
    elif classifier_name in ['resnet18', 'resnet50']:
        classifier = CustomResNet(classifier_name, num_classes=num_classes, use_pretrained=pretrained)

    if checkpoint_path:
        classifier.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    train_transform = classifier.train_transform
    test_transform = classifier.test_transform

    return classifier, train_transform, test_transform

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

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

def get_target_model( target_name, device, domain=None):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    # elif "vit_b" in target_name:
    #     target_name_cap = target_name.replace("vit_b", "ViT_B")
    #     weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
    #     preprocess = weights.transforms()
    #     target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    # elif "resnet" in target_name:
    #     target_name_cap = target_name.replace("resnet", "ResNet")
    #     weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
    #     preprocess = weights.transforms()
    #     target_model = eval("models.{}(weights=weights).to(device)".format(target_name))

    elif target_name == "plumber":

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        convert_models_to_fp32(clip_model)

        if domain == 'spc':
            projector_checkpoint_path = f'logs/classifier/domainnet/plumber/resnet50domain_SPC_lr_0.1_is_mlp_False/best_projector_weights.pth'
        else:    
            # projector_checkpoint_path = f'logs/classifier/imagenet/domainnet/plumber/vit_b_16domain_{domain}_try_lr_0.1_is_mlp_False/projector_weights_final.pth'
            projector_checkpoint_path = f'logs/classifier/domainnet/plumber/resnet50domain_{domain}_lr_0.1_is_mlp_False/best_projector_weights.pth'
            projector_checkpoint_path = f'logs/classifier/cifar10/plumber/SimpleCNNscale_1_epoch1_real_lr_0.1_is_mlp_False/projector_weights_30.pth'
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

def get_data(dataset_name, preprocess=None, domain=None):
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
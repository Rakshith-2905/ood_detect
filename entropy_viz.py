# %%
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils

# from torchcam.methods import LayerCAM, SmoothGradCAMpp
# from torchcam.utils import overlay_mask

import clip

import argparse
import os
import glob
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image


from tqdm import tqdm
from itertools import cycle

from models.resnet import CustomResNet
from models.projector import ProjectionHead
from domainnet_data import DomainNetDataset, get_domainnet_loaders, get_data_from_saved_files
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow, plot_confusion_matrix
from prompts.FLM import generate_label_mapping_by_frequency, label_mapping_base
from models.resnet import CustomClassifier, CustomResNet
import umap
import pickle
# %%
def get_dataset(data_name, domain_name,train_transforms, test_transforms, clip_transform, data_dir='../data'):

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

    return train_dataset, val_dataset, class_names


# %%
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
def get_entropy(val_loader,classifier,clip_model,clip_text_encodings,projector,PROJ_CLIP,Teacher_Temp, device):
    all_clip_embeddings = []

    all_classifier_embeddings = []
    all_proj_embeddings = []
    all_clip_text_embeddings = []
    l = []

    classifier_prob_list, proj_prob_list, CLIP_prob_list = [], [], []
    clip_text_encodings=clip_text_encodings.to(device)

    for i,(images_batch, labels, images_clip_batch) in enumerate(val_loader):
        print(f"batch: {i}")
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)    
        labels = labels.to(device)
        l.append(labels.cpu())
        
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
    l= torch.cat(l, dim=0)
    return classifier_prob_list, proj_prob_list, CLIP_prob_list, l

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



# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
#Checklist
# Did you change the dataset name?
# Did you change the domain name?
# Did you change the scale?
# Did you change the classifier name?
# Did you change the projector weights path?
# did you change num_classes?

data_dir = f"/usr/workspace/KDML/DomainNet"

classifier_name= "resnet50"
num_classes = 345

projector_weights_path= '/usr/workspace/KDML/ood_detect/checkpoints/painting_test_projector/best_projector_weights.pth'
#projector_weights_path = "/usr/workspace/KDML/ood_detect/resnet50_domainnet_real/plumber/resnet50domain_{sketch}_lr_0.1_is_mlp_False/projector_weights_final.pth"
checkpoint_path = f"{data_dir}/best_checkpoint.pth"
PROJ_CLIP = True
dataset_name="domainnet"
domain_name="clipart"

if dataset_name=="domainnet":
    prompt_embeddings_pth = "/usr/workspace/KDML/DomainNet/CLIP_ViT-B-32_text_encodings.pt"
elif dataset_name=="imagenet":
    prompt_embeddings_pth = "/usr/workspace/KDML/ood_detect/CLIP_ViT-B-32_text_encodings_imagenet.pt"

# domainnet_domains_projector= {"real":'/usr/workspace/KDML/ood_detect/checkpoints/real_test_projector/best_projector_weights.pth',\
#                                "sketch": "/usr/workspace/KDML/ood_detect/resnet50_domainnet_real/plumber/resnet50domain_{sketch}_lr_0.1_is_mlp_False/projector_weights_final.pth",\
#                             #   "painting": "/usr/workspace/KDML/ood_detect/checkpoints/painting_test_projector/best_projector_weights.pth",\
#                             #   "clipart": "/usr/workspace/KDML/ood_detect/checkpoints/clipart_test_projector/best_projector_weights.pth",\
#                              #"spc": "/usr/workspace/KDML/ood_detect/checkpoints/SPC_test_projector/best_projector_weights.pth",
# }   
#    
TEACHER_TEMP=2
# domainnet_domains_projector={"real": "/usr/workspace/KDML/ood_detect/checkpoints/teacher_scalling/resnet50scale_100_teT_1_domain_real_lr_0.1_is_mlp_False/best_projector_weights.pth",
#                              "sketch":"/usr/workspace/KDML/ood_detect/checkpoints/teacher_scalling/resnet50scale_100_teT_1_domain_sketch_lr_0.1_is_mlp_False/best_projector_weights.pth"}

# domainnet_domains_projector={"real":"/usr/workspace/KDML/ood_detect/checkpoints/imagenet/vit_b_16_domainnet_real_is_mlp_false.pth",\
#                             "sketch":"/usr/workspace/KDML/ood_detect/checkpoints/imagenet/vit_b_16_domainnet_sketch_is_mlp_false.pth",\
# }
# domainnet_domains_projector={"real": f"/usr/workspace/KDML/ood_detect/checkpoints/batch_size_256/resnet50scale_100_teT_{TEACHER_TEMP}_domain_real_lr_0.1_bs256_is_mlp_False/best_projector_weights.pth",
#                              "sketch": f"/usr/workspace/KDML/ood_detect/checkpoints/batch_size_256/resnet50scale_100_teT_{TEACHER_TEMP}_domain_sketch_lr_0.1_bs256_is_mlp_False/best_projector_weights.pth"\
#                              }

domainnet_domains_projector={"real":f"/usr/workspace/KDML/ood_detect/checkpoints/batch_size_256/all_checkpoints/resnet50scale_100_teT_{TEACHER_TEMP}_domain_real_lr_0.1_bs256_imgweight_0.3_txtweight_0.7_is_mlp_False/best_projector_weights.pth"}

# Load class names from a text file
with open(os.path.join(data_dir, 'class_names.txt'), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
    
####################

    

text_encodings = torch.load(prompt_embeddings_pth)

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

clip_model.eval()
if classifier_name=="resnet50":
    classifier, train_transform, test_transform = build_classifier(classifier_name, num_classes, pretrained=False, checkpoint_path=checkpoint_path)
else:
    classifier, train_transform, test_transform = build_classifier(classifier_name, num_classes, pretrained=True, checkpoint_path=None)
classifier= classifier.to(device)
classifier.eval()

classifier_entropy={}
proj_entropy={}
CLIP_entropy={}

entropy_classifier={}
entropy_proj={}
entropy_CLIP={}

for domain_name in domainnet_domains_projector.keys():

    projector = ProjectionHead(input_dim=512, output_dim=512).to(device)
    projector.load_state_dict(torch.load(domainnet_domains_projector[domain_name])['projector'])
    projector.eval()

    if domain_name=="spc":
       
        datasets_all=[]
        for d_name in ["sketch","clipart","painting"]:

            train_dataset, val_dataset, class_names = get_dataset(dataset_name, d_name,train_transform, test_transform, 
                                                                        data_dir=data_dir, clip_transform=preprocess)
            datasets_all.append(val_dataset)
        
        val_dataset=torch.utils.data.ConcatDataset(datasets_all)
            

    else:

        train_dataset, val_dataset, class_names = get_dataset(dataset_name, domain_name,train_transform, test_transform, 
                                                                    data_dir=data_dir, clip_transform=preprocess)
        

            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    classifier_prob_list, proj_prob_list, CLIP_prob_list, label_list = get_entropy(val_loader,classifier,clip_model,text_encodings,projector,PROJ_CLIP,TEACHER_TEMP,device)
    print(f"CLIP_prob_list: {CLIP_prob_list.shape}", f"proj_prob_list: {proj_prob_list.shape}, classifier_prob_list: {classifier_prob_list.shape}")
    
    entropy_classifier[domain_name] = entropy(classifier_prob_list.cpu().data.numpy())
    entropy_proj[domain_name] = entropy(proj_prob_list.cpu().data.numpy())
    entropy_CLIP[domain_name] = entropy(CLIP_prob_list.cpu().data.numpy())


    label_list=label_list.cpu().numpy()

    classifier_entropy[domain_name] = class_level_entropies(entropy_classifier[domain_name], label_list,num_classes)
    proj_entropy[domain_name] = class_level_entropies(entropy_proj[domain_name], label_list,num_classes)
    CLIP_entropy[domain_name] = class_level_entropies(entropy_CLIP[domain_name], label_list,num_classes)

   

    with open(f"entropy_plots/entropy_data/{domain_name}_{TEACHER_TEMP}_RS_batch_size_{256}_0.3_0.7.pkl", "wb") as f:   
        data={}
        data["classifier_prob_list"]=classifier_prob_list
        data["proj_prob_list"]=proj_prob_list
        data["CLIP_prob_list"]=CLIP_prob_list
        data["label_list"]=label_list


        data["entropy_classifier"]=entropy_classifier[domain_name]
        data["entropy_proj"]=entropy_proj[domain_name]
        data["entropy_CLIP"]=entropy_CLIP[domain_name]
        #classifier_prob_list, proj_prob_list, CLIP_prob_list, label_list
       
        data["classifier_entropy"] = classifier_entropy[domain_name]
        data["proj_entropy"] = proj_entropy[domain_name]
        data["CLIP_entropy"] = CLIP_entropy[domain_name]
        pickle.dump(data, f)
# classifier_entropy={}
# proj_entropy={}
# CLIP_entropy={}
# entropy_classifier_domain={}
# entropy_proj_domain={}
# entropy_CLIP_domain={}
# for domain_name in domainnet_domains_projector.keys():
#     if domain_name!="real":
#         with open(f"entropy_data/{domain_name}.pkl", "rb") as f:
#             data= pickle.load(f)

#         entropy_classifier=data["entropy_classifier"]
#         entropy_proj=data["entropy_proj"]
#         entropy_CLIP=data["entropy_CLIP"]
#         classifier_prob_list=data["classifier_prob_list"]
#         proj_prob_list=data["proj_prob_list"]
#         CLIP_prob_list=data["CLIP_prob_list"]
#         label_list=data["label_list"]
#         label_list=torch.cat(label_list, dim=0).cpu().numpy()

#         entropy_classifier_domain[domain_name]=entropy_classifier
#         entropy_proj_domain[domain_name]=entropy_proj
#         entropy_CLIP_domain[domain_name]=entropy_CLIP


#         classifier_entropy[domain_name] = class_level_entropies(entropy_classifier, label_list,345)
#         proj_entropy[domain_name] = class_level_entropies(entropy_proj, label_list,345)
#         CLIP_entropy[domain_name] = class_level_entropies(entropy_CLIP, label_list,345)
#         print(domain_name)

# %%
# Create a 4x4 grid of subplots
fig, axs = plt.subplots(len(domainnet_domains_projector.keys()), 3, figsize=(12, 12))

#Populate each subplot with a stem plot
for i, domain_name in enumerate(domainnet_domains_projector.keys()):
    
    axs[i, 0].stem(range(num_classes), classifier_entropy[domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
    axs[i, 1].stem(range(num_classes), proj_entropy[domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
    axs[i, 2].stem(range(num_classes), CLIP_entropy[domain_name], basefmt='b', linefmt='r-', markerfmt='ro')
plt.savefig(f"entropy_{TEACHER_TEMP}_correct_RS_bs{256}_0.3_0.7.png")
plt.close()
fig, axs = plt.subplots(len(domainnet_domains_projector.keys()), 3, figsize=(12, 12))
for i, domain_name in enumerate(domainnet_domains_projector.keys()):
    
    axs[i, 0].hist( entropy_classifier[domain_name] )
    axs[i, 1].hist( entropy_proj[domain_name] )
    axs[i, 2].hist( entropy_CLIP[domain_name])


# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.savefig(f"entropy_hist_{TEACHER_TEMP}_correct_RS_bs_{256}_0.3_0.7.png")



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
to_pil = ToPILImage()

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

@torch.no_grad()
def get_embeddings(val_loader,classifier,clip_model,clip_text_encodings,projector,device):
    all_clip_embeddings = []
    all_classifier_embeddings = []
    all_proj_embeddings = []
    all_clip_text_embeddings = []

    clip_text_encodings=clip_text_encodings.to(device)

    for i,(images_batch, labels, images_clip_batch) in enumerate(val_loader):
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)    
        labels = labels.to(device)
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)
        
        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        if PROJ_CLIP: # this is PLUMBER
            proj_embeddings = projector(clip_image_embeddings) # (batch_size, projection_dim)
        else: # this is LIMBER
            proj_embeddings = projector(classifier_embeddings) # (batch_size, projection_dim)
        clip_image_embeddings= F.normalize(clip_image_embeddings, dim=-1)
        proj_embeddings= F.normalize(proj_embeddings, dim=-1)
        classifier_embeddings= F.normalize(classifier_embeddings, dim=-1)
        clip_text_encodings= F.normalize(clip_text_encodings, dim=-1)


        all_clip_text_embeddings.append(clip_text_encodings[labels])
        all_clip_embeddings.append(clip_image_embeddings.detach().cpu())
        all_proj_embeddings.append(proj_embeddings.detach().cpu())
        all_classifier_embeddings.append(classifier_embeddings.detach().cpu())
        if i == 200:
            break


    all_clip_embeddings = torch.cat(all_clip_embeddings, dim=0)
    all_proj_embeddings = torch.cat(all_proj_embeddings, dim=0)
    all_classifier_embeddings = torch.cat(all_classifier_embeddings, dim=0)
    all_clip_text_embeddings = torch.cat(all_clip_text_embeddings, dim=0)
    return all_clip_embeddings, all_proj_embeddings, all_classifier_embeddings,all_clip_text_embeddings

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = f"/usr/workspace/KDML/DomainNet"
prompt_embeddings_pth = "/usr/workspace/KDML/ood_detect/CLIP_ViT-B-32_text_encodings_imagenet.pt"# "/usr/workspace/KDML/DomainNet/CLIP_ViT-B-32_text_encodings.pt"
classifier_name= "vit_b_16" #"resnet50"
num_classes = 1000

projector_weights_path= '/usr/workspace/KDML/ood_detect/checkpoints/painting_test_projector/best_projector_weights.pth'
#projector_weights_path = "/usr/workspace/KDML/ood_detect/resnet50_domainnet_real/plumber/resnet50domain_{sketch}_lr_0.1_is_mlp_False/projector_weights_final.pth"
checkpoint_path = f"{data_dir}/best_checkpoint.pth"
PROJ_CLIP = True
dataset_name="domainnet"
domain_name="clipart"
#domainnet_domains_projector= #{#"real":'/usr/workspace/KDML/ood_detect/checkpoints/real_test_projector/best_projector_weights.pth',\
                              #"sketch": "/usr/workspace/KDML/ood_detect/resnet50_domainnet_real/plumber/resnet50domain_{sketch}_lr_0.1_is_mlp_False/projector_weights_final.pth",\
                             #"painting": "/usr/workspace/KDML/ood_detect/checkpoints/painting_test_projector/best_projector_weights.pth",\
                             #"clipart": "/usr/workspace/KDML/ood_detect/checkpoints/clipart_test_projector/best_projector_weights.pth",        
                             #"spc": "/usr/workspace/KDML/ood_detect/checkpoints/SPC_test_projector/best_projector_weights.pth"}
# domainnet_domains_projector= {"real":'/usr/workspace/KDML/ood_detect/checkpoints/resnet50scale_1_domain_real_lr_0.1_is_mlp_False/best_projector_weights.pth',\
#                                 "sketch": "/usr/workspace/KDML/ood_detect/checkpoints/resnet50scale_1_domain_sketch_lr_0.1_is_mlp_False/best_projector_weights.pth",\
# }
domainnet_domains_projector={"real":"/usr/workspace/KDML/ood_detect/checkpoints/imagenet/vit_b_16_domainnet_real_is_mlp_false.pth",\
                            "sketch":"/usr/workspace/KDML/ood_detect/checkpoints/imagenet/vit_b_16_domainnet_sketch_is_mlp_false.pth",\
}
# Load class names from a text file
with open(os.path.join(data_dir, 'class_names.txt'), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
####################

    

text_encodings = torch.load(prompt_embeddings_pth)

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
if classifier_name=="resnet50":
    classifier, train_transform, test_transform = build_classifier(classifier_name, num_classes, pretrained=False, checkpoint_path=checkpoint_path)
else:
    classifier, train_transform, test_transform = build_classifier(classifier_name, num_classes, pretrained=True, checkpoint_path=None)
classifier= classifier.to(device)
classifier.eval()

clip_embeddings={}
proj_embeddings={}
classifier_embeddings={}
clip_text_embeddings={}
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


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    all_clip_embeddings, all_proj_embeddings, all_classifier_embeddings,all_clip_text_embeddings = get_embeddings(val_loader,classifier,clip_model,text_encodings,projector,device)


    clip_embeddings[domain_name]=all_clip_embeddings
    
    proj_embeddings[domain_name]=all_proj_embeddings
    
    classifier_embeddings[domain_name]=all_classifier_embeddings
    clip_text_embeddings[domain_name]=all_clip_text_embeddings

data={}
with open("clip_image_text_image_all_domain_normalized_embeddings_weight_imagenet_domainnet.pkl","wb") as f:
    data["clip_embeddings"]=clip_embeddings
    data["proj_embeddings"]=proj_embeddings
    data["classifier_embeddings"]=classifier_embeddings
    data["clip_text_embeddings"]=clip_text_embeddings
    pickle.dump(data,f)

# with open("clip_image_text_image_all_domain_normalized_embeddings.pkl","rb") as f:
#     data=pickle.load(f)
# clip_embeddings=data["clip_embeddings"]
# proj_embeddings=data["proj_embeddings"]
# classifier_embeddings=data["classifier_embeddings"]
# clip_text_embeddings=data["clip_text_embeddings"]

text_encodings=F.normalize(text_encodings, dim=-1)

def plot_umap_embeddings(list_tensors,include_lines_for_tensor3=False, labels=None,save_filename=None):
    # Convert PyTorch tensors to NumPy arrays
    tensors_np = [t.cpu().numpy() for t in list_tensors]
    # Combine the embeddings
    combined_embeddings = np.vstack(tensors_np)

    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine')
    embedding_2d = reducer.fit_transform(combined_embeddings)

    # Split the reduced embeddings
    reduced_tensors = np.split(embedding_2d, np.cumsum([len(t) for t in tensors_np])[:-1])

    # Plot the embeddings
    fig, ax = plt.subplots(figsize=(12, 10))
    # create disstinct colors for the len of the list_tensors
    len_tensors = len(list_tensors)
    colors=plt.get_cmap("tab10")
    colors=colors(np.linspace(0,1,len_tensors))
    alphas=[0.5]*len_tensors
    alphas=[1.0]*len_tensors
    alphas[-1]=0.1
    marker_sizes=[10]*len_tensors
    marker_sizes[-1]=40
    marker_shapes=['o']*len_tensors
    marker_shapes[-1]='*'

    for i, reduced_tensor in enumerate(reduced_tensors):
        ax.scatter(reduced_tensor[:, 0], reduced_tensor[:, 1], color=colors[i], label=labels[i],alpha=alphas[i], s=marker_sizes[i], marker=marker_shapes[i])

    # Draw lines between corresponding points for the first two tensors
    # for i in range(len(tensor1_np)):
    #     points = np.vstack((reduced_tensors[0][i], reduced_tensors[1][i]))
    #     ax.plot(points[:, 0], points[:, 1], 'grey', alpha=0.5)

    # # Optionally draw lines for the third tensor
    # if tensor3 is not None and include_lines_for_tensor3 and len(tensor1_np) == len(tensor3_np):
    #     for i in range(len(tensor1_np)):
    #         points = np.vstack((reduced_tensors[0][i], reduced_tensors[2][i]))
    #         ax.plot(points[:, 0], points[:, 1], 'purple', alpha=0.5)

    # Customize the plot with legends
    ax.legend()
    ax.set_title('UMAP projection of the tensor embeddings', fontsize=18)

    plt.savefig(save_filename)



# plot_umap_embeddings(all_clip_embeddings,  all_proj_embeddings,text_encodings.detach().cpu(),labels=['CLIP image', 'Projected image', 'CLIP Text'],save_filename='umap_embeddings.png')

#list_tensor = [clip_embeddings['real'],*proj_embeddings.values(),clip_text_embeddings['real']]
for domain in domainnet_domains_projector.keys():
    list_tensor =[clip_embeddings[domain],proj_embeddings[domain],text_encodings ] #[*proj_embeddings.values()]
    #labels = ['CLIP image', *list(proj_embeddings.keys()), 'CLIP Text']
    labels=  [f'CLIP {domain}', f'Proj {domain}', f'CLIP Text']
    plot_umap_embeddings(list_tensor,labels=labels,save_filename=f'imagenet_umap_embeddings_normalized_clip_proj_text_with_{domain}.png')

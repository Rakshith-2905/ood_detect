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
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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


from train_task_distillation import get_dataset, build_classifier
from models.projector import ProjectionHead
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from models.resnet_cifar import ResNet18
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from models.plumber import PLUMBER
import data_utils.augmix_ops as augmentations
from data_utils.tools import *


@torch.no_grad()
def get_CLIP_text_encodings(clip_model, texts, save_path=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # append "This is a photo of a" to the beginning of each class name
    texts = [f"This is a photo of a {text}" for text in texts]
    with torch.no_grad():
        text_tokens = clip.tokenize(texts)
        text_tokens = fabric.to_device(text_tokens)
        text_encodings = clip_model.encode_text(text_tokens).float()
    # text_encoding_save_path = os.path.join(os.getcwd(), "imagenet_classes_text_encodings.pt")
    torch.save(text_encodings,save_path+'text_encodings.pt')
    return text_encodings

def get_save_dir(args):
    
    projector_name = "plumber" if args.proj_clip else "limber"
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
    

    save_dir = os.path.join(args.save_dir, args.dataset_name, 'tta', projector_name)
    
    save_dir_details = f"{args.prefix}_bs_{args.batch_size}_lr_{args.learning_rate}"
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



def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


def select_confident_samples(logits, topTPT, topAlign):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
    return logits[idxTPT], idxAlign

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def distr_align_loss(out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
    '''
    A feature distibution alignment L1 loss between mean and variance of the features
    '''
    distr_loss = 0
    out_means, out_vars = out_feat
    targ_means, targ_vars = targ_feat
    transf_layers = layers_to
    for l in range(layers_from, transf_layers-1):
        out_mean, out_var = out_means[l], out_vars[l]
        targ_mean, targ_var = targ_means[l], targ_vars[l]
        distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
    return distr_loss





def test_time_tuning(inputs, plumber, args, class_prompts, visual_means, visual_vars):
    
    plumber.set_train_mode()
    selected_idx = None
    inputs = fabric.to_device(inputs)

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            
            output = plumber(inputs, class_prompts) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.tpt_threshold, args.align_threshold)

            loss = avg_entropy(output)

            # Only selected indexes
            # target_feat_distr = (visual_means, visual_vars)
            # out_visual_mean = torch.cat([torch.mean(res.visual_feat[:, selected_idx, :], dim=1, keepdims=True).permute(1,0,2) for res in plumber.clip_model.visual.transformer.resblocks])
            # out_visual_var = torch.cat([torch.mean(((res.visual_feat[:, selected_idx, :] - out_visual_mean[i, :, :].unsqueeze(0).permute(1,0,2))**2), dim=1, keepdims=True).permute(1,0,2) for i, res in enumerate(plumber.clip_model.visual.transformer.resblocks)])
            # out_feat_distr = (out_visual_mean, out_visual_var)

            
            # DISTR_LOSS_W = args.distr_loss_weight / (args.align_layer_to - args.aling_layer_from)
            # loss += DISTR_LOSS_W * distr_align_loss(out_feat_distr, target_feat_distr, layers_from=args.aling_layer_from, layers_to=args.align_layer_to)
        
        
            plumber.zero_grad()
            fabric.backward(loss)
            plumber.optimizer_step()
    return

def test_time_adapt_eval(val_loader, plumber, visual_means, visual_vars, class_prompts, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(len(val_loader),[batch_time, top1, top5], prefix='Test: ')

    for i, (images, target) in enumerate(val_loader):
        start = time.time()
        print(f"TTA for Sample {i+1}")
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.to(device, non_blocking=True)
            image = images
        target = target.to(device, non_blocking=True)
        images = torch.cat(images, dim=0)

        if args.tta_steps > 0:
            with torch.no_grad():
                plumber.set_eval_mode()
                plumber.reset()
        plumber.reset_optimizer()
        test_time_tuning(images, plumber, args, class_prompts, visual_means, visual_vars)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = plumber(image, class_prompts)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - start)
        if (i+1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


def main(args):
    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model.eval()

    plumber = PLUMBER(args.clip_model_name, args.num_classes, img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, dataset_txt_prompt=args.dataset_txt_prompt, 
                      is_mlp=args.is_mlp, device=fabric.device)
    
    plumber = fabric.to_device(plumber)
    

    ########################### Load the dataset ############################
     
    base_transform = transforms.Compose([transforms.Resize(args.resolution, interpolation=BICUBIC),transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    if 'imagenet' in args.dataset_name:
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.n_views-1, augmix=False)
    else:
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.n_views-1, augmix=True)


    # Create the data loader and wrap them with Fabric
    _, val_dataset, class_names = get_dataset(args.dataset_name, data_transform, data_transform, data_dir=args.data_dir, clip_transform=clip_transform)

    # Prompts for every class in the dataset
    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
    
    fabric.print(f"Using {args.dataset_name} dataset")

    # Defining the data loaders
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = fabric.setup_dataloaders(val_loader)

    fabric.print(f"Number of validation examples: {len(val_loader.dataset)}")

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
    
    ####################### Load Imagenet layer statistics computed offline ########
    visual_means = torch.load('/usr/workspace/viv41siv/ICML2024/ood_detect/data_utils/ImgNet_vis_means.pt', map_location='cpu')
    visual_vars = torch.load('/usr/workspace/viv41siv/ICML2024/ood_detect/data_utils/ImgNet_vis_vars.pt', map_location='cpu')
    fabric.to_device(visual_means)
    fabric.to_device(visual_vars)

    ########################### Create the optimizer ############################
    _ = plumber.optimizer_init(args.optimizer, args.learning_rate)
    _ = plumber.scheduler_init()
    fabric.to_device(clip_model)
    
    state = {"clip_model": clip_model,
            "img_projector": plumber.img_projector, "text_projector": plumber.text_projector,
            "clip_prompted_txt_enc": plumber.clip_prompted_txt_enc, "clip_prompted_img_enc": plumber.clip_prompted_img_enc,
            "optimizer_img_proj": plumber.optimizer_img_proj,
            "optimizer_txt_proj": plumber.optimizer_txt_proj, "optimizer_txt_prompt": plumber.optimizer_txt_prompt, 
            "optimizer_img_prompt": plumber.optimizer_img_prompt}
    
    results = test_time_adapt_eval(val_loader, plumber, visual_means, visual_vars, class_prompts, args)
    try:
        print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(args.dataset_name, results[0], results[1]))
    except:
        print("=> Acc. on testset [{}]: {}".format(args.dataset_name, results[0]))
    
    results_dict = {"Dataset": args.dataset_name, "Top1 Accuracy": results[0], "Top5 Accuracy": results[1]}

    fabric.log_dict(results_dict)
    #fabric.save(os.path.join(args.save_dir, "ckpt.pth"), state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--clip_model_name', default='ViT-B/16', help='Name of the CLIP model to use.')
    parser.add_argument('--num_classes', type=int, default=101, help='Number of classes in the dataset')
    parser.add_argument('--img_projection', action='store_true', help='Whether to use task projection or not')
    parser.add_argument('--txt_projection', action='store_true', help='Whether to use text projection or not')
    parser.add_argument('--img_prompting', action='store_true', help='Whether to use image prompting or not')
    parser.add_argument('--cls_txt_prompts', action='store_true', help='Whether to use learnable prompts or not')
    parser.add_argument('--dataset_txt_prompt', action='store_true', help='Whether to use dataset level prompts or class level prompts')
    parser.add_argument('--is_mlp', action='store_true', help='Whether to use MLP projection head or not')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('--n_views', type=int, default=64, help='Total num of views of an image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dataloader')
    parser.add_argument('--data_dir', type=str, default='/usr/workspace/viv41siv/ICASSP2024/LM/LM/', help='Path to the data directory')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='Learning rate for the optimizer')
    parser.add_argument('--tta_steps', type=int, default=1, help='Number of TTA Steps')
    parser.add_argument('--tpt_threshold', type=float, default=0.1, help='Threshold for Entropy')
    parser.add_argument('--align_threshold', type=float, default=0.1, help='Threshold for Alignment')
    parser.add_argument('--distr_loss_weight', type=float, default=100., help='Weight for alignment loss')
    parser.add_argument('--align_layer_to', type=int, default=3, help='Layer uptil which we want to include')
    parser.add_argument('--align_layer_from', type=int, default=0, help='Layer from which we want to include')


    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='ucf101', help='Name of the dataset')    
    parser.add_argument('--n_promt_ctx', type=int, default=16, help='Number of learnable prompt token for each cls')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--proj_clip', action='store_false', help='Whether to project the clip embeddings or the classifier embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')    
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus for DDP per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for DDP')
    parser.add_argument('--template_num', type=int, default=0, help='CLIP text prompt number')

    parser.add_argument('--print_freq', type=int, default=1, help='Print frequency')

    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # Print the arguments
    print(args)
    sys.stdout.flush()
    
    # Make directory for saving results
    args.save_dir = get_save_dir(args)
    args.prompt_path = args.save_dir    
    os.makedirs(os.path.join(args.save_dir, 'lightning_logs'), exist_ok=True)
    
    print(f"\nResults will be saved to {args.save_dir}")
    
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    csv_logger = CSVLogger(args.save_dir, flush_logs_every_n_steps=1)

    fabric = L.Fabric(accelerator=device,num_nodes=args.num_nodes, devices=args.num_gpus, strategy="auto", loggers=[csv_logger])
   
    fabric.launch()

    print = fabric.print

    # The total number of processes running across all devices and nodes
    fabric.print(f"World size: {fabric.world_size}")  # 2 * 3 = 6
            
    seed_everything(args.seed)
    main(args)


'''
python train_eval_tta.py --dataset_name "ucf101" --num_classes 101 --img_projection --data_dir "/usr/workspace/viv41siv/ICASSP2024/LM/LM/" --num_gpus 1 --num_nodes 1
'''





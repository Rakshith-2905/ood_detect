import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

import argparse
import os
from tqdm import tqdm

from domainnet_data import DomainNetDataset
from train_projection_distill_cont import build_classifier
from simple_classifier import SimpleCNN, CIFAR10TwoTransforms

def get_dataset(data_name, train_transforms, test_transforms, clip_transform, data_dir='../data'):
   
    if data_name == 'imagenet':
        train_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_train_examples', transform=train_transforms)
        val_dataset = dset.ImageFolder(root=f'{data_dir}/imagenet_val_examples', transform=test_transforms)
        class_names = train_dataset.classes

    elif data_name == 'domainnet':
        train_dataset = DomainNetDataset(root_dir=data_dir, domain=args.domain_name_name, \
                                        split='train', transform=train_transforms, transform2=clip_transform)
        val_dataset = DomainNetDataset(root_dir=data_dir, domain=args.domain_name_name, \
                                        split='test', transform=test_transforms, transform2=clip_transform)
        class_names = train_dataset.class_names

    elif data_name == 'cifar10':
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform)

        class_names = ['airplane', 'automobile', 'bird']
    elif data_name =="cifar10_full":
        
        train_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=True, transform1=train_transforms, transform2=clip_transform,selected_classes= None)
        val_dataset = CIFAR10TwoTransforms(root=f'{data_dir}/cifar10', train=False, transform1=test_transforms, transform2=clip_transform,selected_classes= None)
        class_names= train_dataset.class_names


    return train_dataset, val_dataset, class_names

def save_features_and_labels(loader, model, clip_model,  device, save_dir, prefix="train", domain="real"):

    model.eval()
    clip_model.eval()
    
    all_outputs = []
    all_features = []
    CLIP_features = []
    all_labels = []

    with torch.no_grad():
        
        pbar = tqdm(loader, desc=f"Saving {prefix} features and labels")
   
        for images_batch, labels, images_clip_batch in pbar:

            images_batch = images_batch.to(device)
            labels = torch.tensor(labels)
            images_clip_batch = images_clip_batch.to(device)

            outputs, features = model(images_batch, return_features=True)

            # Get CLIP image features for the inputs
            image_features = clip_model.encode_image(images_clip_batch)

            CLIP_features.append(image_features.cpu())

            all_outputs.append(outputs.cpu())
            all_features.append(features.cpu())
            all_labels.append(labels)

    # Concatenate the results
    CLIP_features = torch.cat(CLIP_features, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"CLIP features shape: {CLIP_features.shape}")
    print(f"Outputs shape: {all_outputs.shape}")
    print(f"Features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")

    save_dir = os.path.join(save_dir, 'features', domain)
    # Save the results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(CLIP_features, os.path.join(save_dir, f"{prefix}_ViTB32_CLIP_features.pth"))
    torch.save(all_outputs, os.path.join(save_dir, f"{prefix}_outputs.pth"))
    torch.save(all_features, os.path.join(save_dir, f"{prefix}_features.pth"))
    torch.save(all_labels, os.path.join(save_dir, f"{prefix}_labels.pth"))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(args.clip_model_name, device=device)

    model, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    model= model.to(device)
    # Create the data loader and wrap them with Fabric
    print(train_transform,test_transform,preprocess)
    train_dataset, val_dataset, _ = get_dataset(args.dataset_name, train_transform, test_transform,
                                                                data_dir=args.data_dir, clip_transform=preprocess)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get the directory of classifier checkpoints
    save_dir = os.path.dirname(args.classifier_checkpoint_path)

    if not os.path.exists(save_dir):
        assert False, f"Directory {save_dir} does not exist"

    save_features_and_labels(train_loader, model, clip_model, device, save_dir, prefix="train", domain=args.domain_name)
    save_features_and_labels(test_loader, model, clip_model, device, save_dir, prefix="test", domain=args.domain_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--domain_name', type=str, required=True, help='Name of the domain to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to the data directory')


    args = parser.parse_args()
    main(args)


# python script_name.py \
#     --dataset wilds_dataset_name \
#     --domain specific_domain \
#     --image_size 224 \
#     --batch_size 32 \
#     --seed 42 \
#     --classifier_name resnet50 \
#     --num_classes 345 \
#     --classifier_checkpoint_path /path/to/checkpoint.pth \
#     --use_imagenet_pretrained \
#     --clip_model_name ViT-B/32 \
#     --data_dir ./data

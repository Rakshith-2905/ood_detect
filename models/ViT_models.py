import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


import sys
sys.path.append("..")
sys.path.append("../models")

from models.mae import models_mae

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class SAMBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None, device='cuda'):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        """
        self.device = device
        if checkpoint_path is None:
            assert False, "Checkpoint path must be provided for {model_name}"
        elif not os.path.exists(checkpoint_path):
            assert False, f"Checkpoint path does not exist: {checkpoint_path}"

        try:
            self.model = sam_model_registry[model_name](checkpoint=checkpoint_path).to(device)
            # self.predictor = SamPredictor(sam)    
            self.image_encoder = self.model.image_encoder
            # Transform to resize the image to the longest side, add the preprocess that the model expects
            self.transform = transforms.Compose([
                transforms.Resize(1024),
                transforms.ToTensor(),
            ])

        except Exception as e:
            assert False, f"Failed to load SAM model: {e}"
    
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image).to(self.device)
                images_torch.append(self.model.preprocess(image_tensor))

            # images_torch = [self.model.preprocess(self.transform(image).to(self.device)) for image in images]
            images_torch = torch.stack(images_torch) 
        else:
            images_torch = self.model.preprocess(self.transform(images).to(self.device)).unsqueeze(0)
        
        return images_torch

    def forward(self, images):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
                if torch.Tensor is provided, it must be of shape (N, C, H, W) the longer side must be 1024
        Returns:
            torch.Tensor: Features of the image(s) of shape (N, 256,64,64)
        """

        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            assert images.shape[2] == 1024, "The longer side of the image must be 1024"
            images = self.model.preprocess(images.to(self.device))

        features = self.image_encoder(images)
        return features

class MAEBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        """
        if checkpoint_path is None:
            assert False, "Checkpoint path must be provided for {model_name}"
        elif not os.path.exists(checkpoint_path):
            assert False, f"Checkpoint path does not exist: {checkpoint_path}"

        try:
            model = getattr(models_mae, model_name)()
            # load model
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(checkpoint_path, msg))
            # Transform to resize the image to the longest side, add the preprocess that the model expects
            self.transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])
            self.image_encoder = model.forward_encoder
            self.image_decoder = model.forward_decoder
            self.unpatchify = model.unpatchify

        except Exception as e:
            assert False, f"Failed to load model: {e}"
    
    def preprocess_pil(self, images):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)

            images_torch = torch.stack(images_torch) 
        else:
            images_torch = self.transform(images).unsqueeze(0)
        
        return images_torch

    def forward(self, images, decode=False):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)
        else:
            if len(images.shape) == 3:
                images = images.unsqueeze(0)

        features,_, ids_restore = self.image_encoder(images, mask_ratio=0)

        if decode:
            pred = self.image_decoder(features, ids_restore)
            return features, pred

        return features

class DINOBackbone(nn.Module):
    def __init__(self, model_name, checkpoint_path=None ):
        super().__init__()
        """
        Args:
            model_name (str): Name of the model to load from the registry: [vit_h, vit_l, vit_b]
            checkpoint_path (str): Path to the checkpoint file
            device (str): Device to load the model on
        """
        self.device = device

        try:
            model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True)
            self.transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])                
                        ])

        except Exception as e:
            assert False, f"Failed to load model: {e}"
    
    def preprocess_pil(self, images, image_size=224, image_crop_size=224):
        # check if images is a list, then preprocess each image
        if isinstance(images, list):
            images_torch = []
            for image in images:
                image_tensor = self.transform(image)
                images_torch.append(image_tensor)

            images_torch = torch.stack(images_torch) 

        else:
            images_torch = self.transform(images).unsqueeze(0)
        
        return images_torch

    def forward(self, images):
        """
        Args:
            images (pil image(s) or torch.Tensor): Image(s) to extract features from:
                if pil image(s) are provided, they will be converted to torch.Tensor
        """
        if not isinstance(images, torch.Tensor):
            images = self.preprocess_pil(images)

        features = self.model(images)
        return features

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sam = SAMBackbone(model_name="vit_h", checkpoint_path="checkpoints/sam_vit_h_4b8939.pth").to(device)
    # mae = MAEBackbone(model_name="mae_vit_large_patch16", checkpoint_path='./checkpoints/mae_visualize_vit_large_ganloss.pth').to(device)
    dino = DINOBackbone(model_name="dino_vits16", checkpoint_path=None).to(device)

    pil_image = Image.open("../data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    pil_images = [pil_image, pil_image, pil_image, pil_image, pil_image, pil_image, pil_image, pil_image]

    torch_images = torch.randn(4, 3, 224, 256).to(device)

    with torch.no_grad():
        # features = sam(pil_images)
        # features, y = mae(pil_images, decode=True)
        features = dino(pil_images)

    print(features.shape)

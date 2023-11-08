import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import sys
sys.path.append("..")
# from utils import ResizeLongestSide

import os
import numpy as np
import cv2
from PIL import Image
# from trans import ResizeLongestSide


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

if __name__ == "__main__":

    model_name = "vit_h"
    checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
    device = "cuda"
    sam = SAMBackbone(model_name, checkpoint_path, device)

    np_image = cv2.imread("./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    pil_image = Image.open("./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg")
    pil_images = [pil_image, pil_image]
    torch_images = torch.randn(4, 3, 224, 256).to(device)

    with torch.no_grad():
        features = sam(pil_images)
        print(features.shape)

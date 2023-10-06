import torch
from torchcam.methods import LayerCAM, SmoothGradCAMpp
import torchvision
import torch.nn.functional as F
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToPILImage

import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
from random import sample

from dataloader import WildsDataLoader


to_pil = ToPILImage()

def generate_masked_image(img_tensor, cam_extractor, predicted_probs, target_class, 
                          threshold=0.8, save=True, content_mask=True, use_pred_cls=False):
    """
    Generate a masked image using CAM for a given tensor batch and optionally overlay the CAM heatmap on the original image.

    Parameters:
    - img_tensor (torch.Tensor): A batch of images. Tensor shape should be BxCxHxW and preprocessed (e.g., normalized).
    - cam_extractor (torchcam extractor): A TorchCAM extractor object to extract the CAM from the model.
    - predicted_probs (torch.Tensor): The predicted probabilities from the model.
    - target_class (list): List of target class indices, one for each image in the batch.
    - threshold (float, optional): Threshold to binarize the CAM and create the mask. Defaults to 0.8.
    - save (bool, optional): If True, saves the resulting masked images to disk and overlays the CAM heatmap on the original images. Defaults to False.
    - content_mask (bool, optional): If True, inverts the CAM values such that 1 becomes 0 and 0 becomes 1. Defaults to False.
    - use_pred_cls (bool, optional): If True, uses the predicted class instead of the target class. Defaults to False.

    Returns:
    - torch.Tensor: A batch of masked images.

    Note:
    If `save` is True, the images with the CAM heatmap overlay are saved as "cam_overlay_i.png", where `i` is the index 
    of the image in the batch. The corresponding masked images are saved as "masked_image_i.png".
    """

    device = img_tensor.device

    masked_imgs = []

    target_class_list = [c.item() for c in target_class]

    if use_pred_cls:
        target_class_list = [torch.argmax(p).item() for p in predicted_probs]

    cams = cam_extractor(target_class_list, predicted_probs)[0]

    orig_img = img_tensor.clone()
    for i, img in enumerate(img_tensor):

        # Resize the CAM to the prompted image size
        resized_tensor = F.interpolate(cams[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True)
        
        # grid = torch.stack([resized_tensor[0].repeat(3, 1, 1)])
        # grid = torchvision.utils.make_grid(grid, nrow=1, normalize=True, range=(-1, 1))


        # torchvision.utils.save_image(grid, f"cam_img_invt_{i}.png")
        
        # Optionally, squeeze out the added dimensions to get [224, 224]
        cam_resized_orig = resized_tensor.squeeze(0).squeeze(0)
        
        if content_mask:
            mask = (cam_resized_orig < threshold).to(dtype=torch.float32)
        else:
            mask = (cam_resized_orig > threshold).to(dtype=torch.float32)
        
        masked_img = img_tensor[i] * mask[None, :, :]
        masked_imgs.append(masked_img)

        if save:

            # Overlay the cam to the prompted image as a heatmap
            cam_overlay = overlay_mask(to_pil_image(orig_img[i]), to_pil_image(cam_resized_orig, mode='F'), alpha=0.5)
            # Convert to tensor
            cam_overlay = torchvision.transforms.ToTensor()(cam_overlay).to(device)

            # make make to 3 channels
            mask = mask.repeat(3, 1, 1)

            resized_tensor = resized_tensor[0].repeat(3, 1, 1)
            
            grid = torch.stack([orig_img[i], cam_overlay, resized_tensor, mask, masked_img])
            grid = torchvision.utils.make_grid(grid, nrow=5, normalize=True, range=(-1, 1))


            torchvision.utils.save_image(grid, f"img_invt_{i}.png")

    return torch.stack(masked_imgs)

def main(args):
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = torchvision.models.resnet18()  # You can change this to your specific model type
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_set.selected_classes))  # Adjusting the final layer based on your dataset
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    # Load the validation set using WildsDataLoader
    val_set = WildsDataLoader(dataset_name=args.dataset, split="val", image_size=args.image_size, batch_size=args.batch_size)
    val_loader = val_set.load_data()

    # Randomly select 10 images from validation set
    random_images = sample(list(val_loader), 10)

    # Create a CAM extractor
    cam_extractor = SmoothGradCAMpp(model)

    # Compute the CAMs for these 10 images and display
    for inputs, labels, _ in random_images:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get model's predictions
        outputs = model(inputs)
        predicted_probs = F.softmax(outputs, dim=1)

        # Generate the masked image
        generate_masked_image(inputs, cam_extractor, predicted_probs, labels, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRAD-CAMs on the validation dataset using the trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    args = parser.parse_args()

    main(args)
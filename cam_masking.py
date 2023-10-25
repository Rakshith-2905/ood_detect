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
from sklearn.manifold import TSNE

import argparse
import glob
import os
from random import sample
from tqdm import tqdm
from itertools import cycle

from dataloader import WildsDataLoader
from models.resnet import CustomResNet


to_pil = ToPILImage()

def plot_classwise_tsne(model, val_loader, val_novel_loader, selected_classes, selected_novel_classes, device, save_path):
    """
    Compute, plot, and save t-SNE embeddings of features extracted from a model for selected classes and overlay novel samples.

    Parameters:
    - model: The trained model with a method forward_features to extract features.
    - val_loader: DataLoader for the validation set.
    - val_novel_loader: DataLoader for the novel set to be overlayed in gray.
    - selected_classes: List of selected class IDs.
    - selected_novel_classes: List of selected novel class IDs.
    - device: Torch device ('cuda' or 'cpu').
    - save_path: Path to save the generated plot.
    """
    features_list = []
    labels_list = []

    def extract_features(loader, selected_classes, is_novel=False, max_samples=200):
        samples_per_class = {cls: 0 for cls in selected_classes}
        for inputs, labels, _ in tqdm(loader):
            with torch.no_grad():
                selected_indices = []
                
                # For each class, choose a subset of indices, ensuring we don't exceed max_samples
                for cls in selected_classes:
                    cls_indices = (labels == cls).nonzero(as_tuple=True)[0].cpu().numpy()
                    available_slots = max_samples - samples_per_class[cls]
                    
                    if available_slots > 0:
                        chosen_indices = np.random.choice(cls_indices, min(len(cls_indices), available_slots), replace=False)
                        samples_per_class[cls] += len(chosen_indices)
                        selected_indices.extend(chosen_indices)

                # If it's the novel set, modify max samples
                if is_novel:
                    max_samples = 100

                selected_inputs = inputs[selected_indices].to(device)

                if len(selected_inputs) == 0:
                    continue

                features = model(selected_inputs, return_features=True).cpu()
                features_list.append(features)
                
                if is_novel:
                    labels_list.append(torch.full((len(features),), -1))  # Assign label -1 for novel samples
                else:
                    labels_list.append(labels[selected_indices].cpu())
                
                # Clear CUDA cache
                del selected_inputs
                torch.cuda.empty_cache()

    # Extract features from both loaders
    extract_features(val_loader, selected_classes)
    extract_features(val_novel_loader, selected_novel_classes, is_novel=True)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    features_np = features.numpy()
    labels_np = labels.numpy()

    tsne = TSNE(n_components=2, random_state=42)
    features_2D = tsne.fit_transform(features_np)

    # Create a list of distinct and vivid markers and colors.
    markers = cycle(('o', 's', '^', 'v', 'p', '*', 'H', '+', 'x', 'D', '|'))
    colors = cycle(plt.cm.tab20.colors)  # This colormap has 20 distinct colors.
    
    plt.figure(figsize=(12, 12))
    for i, class_id in enumerate(selected_classes):
        color = next(colors)
        marker = next(markers)
        
        indices = np.where(labels_np == class_id)
        
        # Use the distinct color and marker for each class.
        plt.scatter(features_2D[indices, 0], features_2D[indices, 1], label=str(class_id), 
                    color=color, edgecolor='k', s=25, alpha=0.6, marker=marker)
    
    novel_indices = np.where(labels_np == -1)
    plt.scatter(features_2D[novel_indices, 0], features_2D[novel_indices, 1], color='gray', s=10, label='Novel', alpha=0.5)
    
    plt.legend(loc='best')
    plt.title("t-SNE Visualization of Feature Vectors with Overlayed Novel Samples")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def generate_masked_image(img_tensor, cam_extractor, predicted_probs, target_class, 
                          pred_classes_list, gt_classes_list, threshold=0.8, 
                          save=True, content_mask=True, use_pred_cls=False):
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
    - pred_classes_list (list): List of class names corresponding to the predicted class indices.
    - gt_classes_list (list): List of class names corresponding to the gt class indices.

    Returns:
    - torch.Tensor: A batch of masked images.
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
        cam_resized_orig = resized_tensor.squeeze(0).squeeze(0)
        
        if content_mask:
            mask = (cam_resized_orig < threshold).to(dtype=torch.float32)
        else:
            mask = (cam_resized_orig > threshold).to(dtype=torch.float32)
        
        masked_img = img_tensor[i] * mask[None, :, :]
        masked_imgs.append(masked_img)

        # Get predicted class name and gt class name
        pred_cls_name = pred_classes_list[torch.argmax(predicted_probs[i]).item()]
        gt_cls_name = gt_classes_list[target_class[i].item()]

        if save:

            # Overlay the cam to the prompted image as a heatmap
            cam_overlay = overlay_mask(to_pil_image(orig_img[i]), to_pil_image(cam_resized_orig, mode='F'), alpha=0.5)
            # Convert to tensor
            cam_overlay = torchvision.transforms.ToTensor()(cam_overlay).to(device)

            # make mask to 3 channels
            mask = mask.repeat(3, 1, 1)
            resized_tensor = resized_tensor[0].repeat(3, 1, 1)
            
            grid = torch.stack([orig_img[i], cam_overlay, masked_img])
            grid = torchvision.utils.make_grid(grid, nrow=3, normalize=True, range=(-1, 1))

            # Modify the save_image name to include predicted and gt class names
            torchvision.utils.save_image(grid, f"img_pred_{pred_cls_name}_{i}.png")

    return torch.stack(masked_imgs)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    train_set = WildsDataLoader(dataset_name=args.dataset, split="train", image_size=args.image_size, batch_size=args.batch_size, class_percentage=args.class_percentage, seed=args.seed)
    train_loader = train_set.load_data()
    train_set.display_details()
    train_classes = train_set.selected_classes

    val_set= WildsDataLoader(dataset_name=args.dataset, split="val", image_size=args.image_size, batch_size=args.batch_size, class_percentage=0.5, selected_classes=train_set.selected_classes, use_train_classes=True)
    val_loader = val_set.load_data()
    val_set.display_details()
    val_classes = val_set.selected_classes

    val_novel_set= WildsDataLoader(dataset_name=args.dataset, split="val", image_size=args.image_size, batch_size=args.batch_size, class_percentage=0.5, selected_classes=train_set.selected_classes, use_train_classes=False)
    val_novel_loader = val_novel_set.load_data()
    val_novel_set.display_details()
    val_novel_classes = val_novel_set.selected_classes

    # Compare the selected classes in the train and validation sets if they dont match assert
    assert (train_classes == val_classes).all(), "Selected classes in train and validation sets do not match"

    # Compare the val classes and val novel classes, if any of the val classes are in val novel classes assert
    assert len(np.intersect1d(val_classes, val_novel_classes)) == 0, "Val classes and val novel classes have an overlap"



    model = CustomResNet(model_name=args.resnet_model, num_classes=len(train_set.selected_classes))
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    model_dir = os.path.dirname(args.model_path)
    print(f"\n{args.resnet_model} model with '{args.model_path}' weights loaded successfully.")

    # Randomly select 10 integers from 0 to len(train_set.selected_classes)
    random_classes = sample(range(len(train_set.selected_classes)), 10)
    random_novel_classes = sample(range(len(val_novel_classes)), 10)

    # plot_classwise_tsne(model, val_loader, val_novel_loader, random_classes, random_novel_classes, device, save_path=os.path.join(model_dir, 'tsne.png'))

    domainnet_categories = [os.path.basename(path) for path in glob.glob('data/domainnet_v1.0/real/*')]
    domainnet_categories = sorted(domainnet_categories, key=lambda x: x.lower())

    # fet the train class names
    train_classes_list = [domainnet_categories[i] for i in train_classes]
    # Save the train classes list
    with open(os.path.join(model_dir, 'train_classes.txt'), 'w') as f:
        for item in train_classes_list:
            f.write("%s\n" % item)
    print(train_classes_list)
    assert False
    val_classes_list = [domainnet_categories[i] for i in val_novel_classes]

    print("Creating CAM extractor...")
    # Create a CAM extractor
    cam_extractor = SmoothGradCAMpp(model)

    # Compute the CAMs for these 10 images and display
    for inputs, labels, _ in tqdm(val_novel_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get model's predictions
        outputs = model(inputs)
        predicted_probs = F.softmax(outputs, dim=1)

        # Generate the masked image
        generate_masked_image(inputs, cam_extractor, predicted_probs, labels, train_classes_list, domainnet_categories, threshold=0.5, save=True, content_mask=False, use_pred_cls=False)
        assert False
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRAD-CAMs on the validation dataset using the trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--resnet_model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18', help='Type of ResNet model to use')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=0.5, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    main(args)
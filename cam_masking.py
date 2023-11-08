import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import ToPILImage
import torchvision.utils as vutils

from torchcam.methods import LayerCAM, SmoothGradCAMpp
from torchcam.utils import overlay_mask

import clip

import argparse
import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image


from tqdm import tqdm
from itertools import cycle

from models.resnet import CustomResNet
from models.projector import ProjectionHead
from domainnet_data import DomainNetDataset, get_domainnet_loaders, get_data_from_saved_files
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow, plot_confusion_matrix
from prompts.FLM import generate_label_mapping_by_frequency, label_mapping_base


to_pil = ToPILImage()



def unnormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # mean=[0.48145466, 0.4578275, 0.40821073]
    # std=[0.26862954, 0.26130258, 0.27577711]
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std_tensor[:, None, None]).add_(mean_tensor[:, None, None])
    return tensor

def save_image(tensor, file_name):

    tensor = tensor.detach().cpu()
    # Ensure it's in the range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to image and save
    vutils.save_image(tensor, file_name)

def generate_masked_image(img_tensor, cam_extractor, predicted_probs, target_class, 
                          class_names, threshold=0.8, 
                          save=True, content_mask=True, use_pred_cls=True, save_dir='masked_images'):
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

    # masked_imgs = []

    # target_class_list = [c.item() for c in target_class]

    # if use_pred_cls:
    #     target_class_list = [torch.argmax(p).item() for p in predicted_probs]

    # cams = cam_extractor(target_class_list, predicted_probs)[0]

    orig_img = img_tensor.clone()
    for i, img in enumerate(img_tensor):

    #     # Resize the CAM to the prompted image size
    #     resized_tensor = F.interpolate(cams[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=True)
    #     cam_resized_orig = resized_tensor.squeeze(0).squeeze(0)
        
    #     if content_mask:
    #         mask = (cam_resized_orig < threshold).to(dtype=torch.float32)
    #     else:
    #         mask = (cam_resized_orig > threshold).to(dtype=torch.float32)
        
    #     masked_img = img_tensor[i] * mask[None, :, :]
    #     masked_imgs.append(masked_img)

    #     # Get predicted class name and gt class name
    #     pred_cls_name = class_names[torch.argmax(predicted_probs[i]).item()]
    #     gt_cls_name = class_names[target_class[i].item()]

        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # # Overlay the cam to the prompted image as a heatmap
            # cam_overlay = overlay_mask(to_pil_image(orig_img[i]), to_pil_image(cam_resized_orig, mode='F'), alpha=0.5)
            # # Convert to tensor
            # cam_overlay = torchvision.transforms.ToTensor()(cam_overlay).to(device)

            # # make mask to 3 channels
            # mask = mask.repeat(3, 1, 1)
            # resized_tensor = resized_tensor[0].repeat(3, 1, 1)
            
            # unnormalized_masked_img = unnormalize(masked_img.clone())  # Clone to avoid modifying in-place
            # save_image(unnormalized_masked_img, f"{save_dir}/masked_{i}.png")

            # Repeat for orig_img[i]
            unnormalized_orig_img = unnormalize(orig_img[i].clone())
            save_image(unnormalized_orig_img, f"{save_dir}/orig_{i}.png")


            # masked_img = torch.stack([masked_img])

            # masked_img = torchvision.utils.make_grid(masked_img, nrow=1, normalize=True, range=(0, 1))

            # # Modify the save_image name to include predicted and gt class names
            # torchvision.utils.save_image(masked_img, f"{save_dir}/masked_{i}.png")
            
            # orig_img_ = torch.stack([orig_img[i]])
            # orig_img_ = torchvision.utils.make_grid(orig_img_, nrow=1, normalize=True, range=(0, 1))

            # # Modify the save_image name to include predicted and gt class names
            # torchvision.utils.save_image(orig_img_, f"{save_dir}/orig_{i}.png")


    # return torch.stack(masked_imgs)

def load_image(file_path):
    """
    Load an image and convert it to a NumPy array with values in the range [0, 255].

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Image as a NumPy array with values in the range [0, 255].
    """
    # Open the image file
    image = Image.open(file_path)

    # Convert to RGB mode if not already in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to NumPy array
    image_array = np.array(image)

    # Ensure values are in the range [0, 255]
    image_array = np.clip(image_array, 0, 255)

    return image_array

def main(args):

    base_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.eval()

    not_CLIP_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    # Load class names from a text file
    with open(os.path.join(args.data_dir, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    loaders, _ = get_domainnet_loaders("real", batch_size=args.batch_size, data_dir=args.data_dir)
    
    train_loader = loaders['train']
    val_loader = loaders['test']


    # Load your trained model from checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    
    resnet_model = CustomResNet(model_name=args.resnet_model, num_classes=345)
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    resnet_model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    resnet_model.to(device)


    projector = ProjectionHead(input_dim=args.resnet_dim, output_dim=args.projection_dim).to(device)
    # Load projector weights from checkpoint
    projector.load_state_dict(torch.load(args.projector_checkpoint_path))
    print(f"Loaded projector weights from {args.projector_checkpoint_path}")
    projector.eval()

    label_mapping = None

    text_encodings = torch.load(args.prompt_embeddings_pth)[0]

    im = load_image('./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg')
    
    transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    # image = transform(Image.fromarray(im.astype(np.uint8))).unsqueeze(0)
    image = transform(Image.open('./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg')).unsqueeze(0)

    l = torch.from_numpy(np.array([317]))
    valset = torch.utils.data.TensorDataset(image, l)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

    # Preprocess the image for clip
    image_CLIP = preprocess(Image.open('./data/domainnet_v1.0/real/toothpaste/real_318_000284.jpg')).unsqueeze(0).to(device)
    # Encode the image using CLIP encoder_image
    orig_img_feature = clip_model.encode_image(image_CLIP)

    # Compute similarities between image embeddings and text encodings
    orig_clip_similarities = compute_similarities(orig_img_feature, text_encodings, mode="cosine")
    orig_clip_prob = F.softmax(orig_clip_similarities, dim=-1)
    orig_clip_predictions = torch.argmax(orig_clip_prob, dim=-1)

    print(f"Loaded text encodings of shape: {text_encodings.shape}")

    # save directory is the director of the projector checkpoint
    save_dir = os.path.dirname(args.projector_checkpoint_path)
    save_dir = os.path.join(save_dir, 'GradCAM_masked_images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    resnet_model.eval()
    projector.eval()
    
    # print("Creating CAM extractor...")
    # # Create a CAM extractor
    # cam_extractor = SmoothGradCAMpp(resnet_model)

    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating')
    
    for images, labels in pbar:

        images, labels = images.to(device), labels.to(device)

        for i in range(len(images)):
            # Preprocess the image for clip
            unnormalized_orig_img = unnormalize(images[i].clone())
            save_image(unnormalized_orig_img, f"{save_dir}/orig_{i}.png")

        # Get the ResNet predictions
        resnet_logits, resnet_embeddings = resnet_model(images, return_features=True)
        probs_from_resnet = F.softmax(resnet_logits, dim=-1)
        resnet_predictions = torch.argmax(probs_from_resnet, dim=-1)
        
        # Project the resnet embeddings
        proj_embeddings = projector(resnet_embeddings)
        
        # Print the min and max values of the projected embeddings
        print(f"Range of projected embeddings: {torch.min(proj_embeddings)}, {torch.max(proj_embeddings)}")
        print(f"Range of text encodings: {torch.min(text_encodings[317])}, {torch.max(text_encodings[317])}")

        # Compute the predictions using the projected embeddings
        similarities = compute_similarities(proj_embeddings, text_encodings, mode=args.similarity_mode)
        probs_from_proj = F.softmax(similarities, dim=-1)
        proj_predictions = torch.argmax(probs_from_proj, dim=-1)

        # # Generate the masked image
        # masked_images = generate_masked_image(images, cam_extractor, probs_from_resnet, labels, class_names, threshold=0.15, save=True, content_mask=False, use_pred_cls=True, save_dir=save_dir)
        
        # Load the images from the save directory and compute CLIP features
        # masked_CLIP_features = []
        reloaded_CLIP_features = []
        orig_CLIP_features = []
        # for i, saved_img_pth in enumerate(glob.glob(f"{save_dir}/*.png")):
        image_paths = [f"{save_dir}/orig_{i}.png" for i in range(len(images))]
        for i, saved_img_pth in enumerate(image_paths):
            reloaded_image = preprocess(Image.open(saved_img_pth)).unsqueeze(0).to(device)

            with torch.no_grad():
                reloaded_image_features = clip_model.encode_image(reloaded_image)
                # orig_img_feature = clip_model.encode_image(images[i].unsqueeze(0).to(device))

                reloaded_CLIP_features.append(reloaded_image_features)
                # orig_CLIP_features.append(orig_img_feature)
                # if "test_" in saved_img_pth:
                #     CLIP_img_features.append(image_features)
                # else:
                #     masked_CLIP_features.append(image_features)
        
        # masked_CLIP_features = torch.cat(masked_CLIP_features, dim=0)
        reloaded_CLIP_features = torch.cat(reloaded_CLIP_features, dim=0)
        # orig_CLIP_features = torch.cat(orig_CLIP_features, dim=0)


        # Print the min and max values of the CLIP features
        print(f"Range of CLIP features: {torch.min(reloaded_CLIP_features)}, {torch.max(reloaded_CLIP_features)}")
        assert False

        # # Compute similarities between masked image embeddings and text encodings
        # masked_similarities = compute_similarities(masked_CLIP_features, text_encodings, mode=args.similarity_mode)
        # prob_masked_from_proj = F.softmax(masked_similarities, dim=-1)
        # masked_predictions = torch.argmax(prob_masked_from_proj, dim=-1)

        # # Compute similarities between original image embeddings and text encodings
        # orig_clip_similarities = compute_similarities(orig_CLIP_features, text_encodings, mode="cosine")
        # orig_clip_prob = F.softmax(orig_clip_similarities, dim=-1)
        # orig_clip_predictions = torch.argmax(orig_clip_prob, dim=-1)

        # Compute similarities between original image embeddings and text encodings
        reloaded_clip_similarities = compute_similarities(reloaded_CLIP_features, text_encodings, mode="cosine")
        reloaded_clip_prob = F.softmax(reloaded_clip_similarities, dim=-1)
        reloaded_clip_predictions = torch.argmax(reloaded_clip_prob, dim=-1)

        # Print the predictions and the probabilities
        for i in range(len(images)):
            print(f"\nResNet prediction: {resnet_predictions[i]} ({probs_from_resnet[i][resnet_predictions[i]]})")
            print(f"Projected prediction: {proj_predictions[i]} ({probs_from_proj[i][proj_predictions[i]]})")
            # print(f"Masked Zero-shot prediction: {masked_predictions[i]} ({prob_masked_from_proj[i][masked_predictions[i]]})")
            print(f"Reloaded zero-shot prediction: {reloaded_clip_predictions[i]} ({reloaded_clip_prob[i][reloaded_clip_predictions[i]]})")
            print(f"Original Zero-shot prediction: {orig_clip_predictions[i]} ({orig_clip_prob[i][orig_clip_predictions[i]]})")
            print(f"Label: {labels[i]}")
        assert False
        # L2 norm of the embeddings

        # # Save the embeddings as numpy arrays
        # np.save(f"{save_dir}/CLIP_img_features.npy", CLIP_img_features[0].detach().cpu().numpy())
        # np.save(f"{save_dir}/masked_CLIP_features.npy", masked_CLIP_features[0].detach().cpu().numpy())
        # np.save(f"{save_dir}/proj_embeddings.npy", proj_embeddings[0].detach().cpu().numpy())

        # CLIP_img_features_norm = torch.norm(CLIP_img_features, dim=-1)
        # masked_CLIP_features_norm = torch.norm(masked_CLIP_features, dim=-1)
        # proj_embeddings_norm = torch.norm(proj_embeddings, dim=-1)

        # print(f"Original CLIP features norm: {CLIP_img_features_norm}")
        # print(f"Masked CLIP features norm: {masked_CLIP_features_norm}")
        # print(f"Projected embeddings norm: {proj_embeddings_norm}")

        # # Compute similarities between masked image embeddings and projected embeddings
        # masked_proj_similarities = compute_similarities(masked_CLIP_features, proj_embeddings, mode="cosine").detach().cpu().numpy()
        # orig_proj_similarities = compute_similarities(CLIP_img_features, proj_embeddings, mode="cosine").detach().cpu().numpy()
        # masked_orig_similarities = compute_similarities(masked_CLIP_features, CLIP_img_features, mode="cosine").detach().cpu().numpy()

        # for i in range(len(images)):
        #     print(f"\nImage_{i}\n")
        #     print(f"CLIP Original vs. Projected similarity: {orig_proj_similarities[i]}")
        #     print(f"CLIP Masked vs. Projected similarity: {masked_proj_similarities[i]}")
        #     print(f"CLIP Masked vs. Original similarity: {masked_orig_similarities[i]}")

        assert False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRAD-CAMs on the validation dataset using the trained model.")

    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
    parser.add_argument('--data_dir', type=str, default='data/domainnet_v1.0', help='Path to the WILDS dataset')
    parser.add_argument('--domain', type=str, required=True, help='Name of the domain to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=1, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--resnet_model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18', help='Type of ResNet model to use')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--projector_checkpoint_path', type=str, help='Path to projector checkpoint to resume training from')

    parser.add_argument('--resnet_dim', type=int, default=2048, help='Dimension of the ResNet embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--prompt_embeddings_pth', type=str, required=True, help='Path to the prompt embeddings')
    parser.add_argument('--use_default_prompt', type=bool, default=True, help='Use the default prompt instead of FLM')
    parser.add_argument('--mapping_num', type=int, default=1, help='Number of labels to map to each prompt')
    parser.add_argument('--similarity_mode', type=str, choices=['cosine', 'DN', 'DN*'], default='cosine', help='Type of similarity to use')
    
    args = parser.parse_args()

    args = parser.parse_args()

    main(args)
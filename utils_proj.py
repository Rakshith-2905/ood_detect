import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms import Normalize, Resize
import random

from data_utils.cifar10_data import get_CIFAR10_dataloader


class SimpleDINOLoss(nn.Module):
    def __init__(self, student_temp=0.1, teacher_temp=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # Teacher sharpening
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        # teacher_out = teacher_out.detach()

        # CE(p, q) = -sum_{i} p_i * log(q_i)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()

def compute_accuracy(probs, labels):
    predictions = probs.argmax(dim=1)
    correct = (predictions == labels).float().sum()
    return (correct / probs.size(0)).item()

def compute_similarities(image_embeddings, text_embeddings, mode='cosine', logits_scale=100):
    if mode == 'cosine':
        return cosine_similarities(image_embeddings, text_embeddings)
    # TODO: add mean for DN
    elif mode == 'DN':
        return CLIP_DN_similarities(image_embeddings, text_embeddings)
    elif mode == 'DN*':
        cos_sim = cosine_similarities(image_embeddings, text_embeddings)
        dn_sim = CLIP_DN_similarities(image_embeddings, text_embeddings)
        return (cos_sim + dn_sim)/2

def cosine_similarities(image_embeddings, text_embeddings, logits_scale=100):
    """ Compute cosine similarities between image embeddings and text encodings for all labels """
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    # make the text embeddings to the same data type as image embeddings
    text_embeddings = text_embeddings.type_as(image_embeddings)
    similarities = logits_scale*F.cosine_similarity(image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=2)
    
    return similarities

# load the image and text embeddings from the saved files
def load_embeddings(save_dir='prompts/'):
    mean_image_embeddings = torch.load(os.path.join(save_dir, "RN50_mean_image_embeddings.pth"))
    mean_text_embeddings = torch.load(os.path.join(save_dir, "RN50_mean_text_embeddings.pth"))

    return mean_image_embeddings, mean_text_embeddings

def CLIP_DN_similarities(image_embeddings, text_embeddings):
    "Compute cos similarity with distribution normalization"

    # Compute the mean of the embeddings
    mean_image_embeddings, mean_text_embeddings = load_embeddings()

    DN_image_embeddings = image_embeddings - mean_image_embeddings.unsqueeze(0)/2
    DN_text_embeddings = text_embeddings - mean_text_embeddings.unsqueeze(0)/2
    
    similarities = compute_similarities(DN_image_embeddings, DN_text_embeddings)

    return similarities
 

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Args:
        named_parameters: model.named_parameters(), list of tuple containing name and parameters
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if p.grad is None:
            print(f"No gradient for: {n}")
            assert False
        elif p.requires_grad and ("bias" not in n):  # Check if gradient is not None
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

# This function assumes you have the true labels and predictions as 1D numpy arrays.
def plot_confusion_matrix(proj_labels, resnet_labels, class_names, save_dir=None ):
    
    # Compute the confusion matrix
    cm = confusion_matrix(resnet_labels, proj_labels, normalize='true')

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the size as needed
    sns.heatmap(cm, annot=False, ax=ax, cmap='Blues', cbar=True)
    
    # Labels and title
    ax.set_xlabel('Projected Predictions', fontsize=12)
    ax.set_ylabel('ResNet Predictions', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=15)

    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

def plot_umap_embeddings(tensor1, tensor2, tensor3=None, include_lines_for_tensor3=False, labels=None):
    # Convert PyTorch tensors to NumPy arrays
    tensor1_np = tensor1.detach().cpu().numpy()
    tensor2_np = tensor2.detach().cpu().numpy()
    
    tensors_np = [tensor1_np, tensor2_np]
    
    # Include the third tensor if it's provided
    if tensor3 is not None:
        tensor3_np = tensor3.detach().cpu().numpy()
        tensors_np.append(tensor3_np)

    # Combine the embeddings
    combined_embeddings = np.vstack(tensors_np)

    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine')
    embedding_2d = reducer.fit_transform(combined_embeddings)

    # Split the reduced embeddings
    reduced_tensors = np.split(embedding_2d, np.cumsum([len(t) for t in tensors_np])[:-1])

    # Plot the embeddings
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['red', 'blue', 'green']
    alphas=[0.1,0.1,1.0]
    marker_sizes=[10,10,40]
    marker_shapes=['o','o','*']

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

    plt.savefig('umap_embeddings.png')

class ImageTransforms:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], k=3, image_size=None, specific_transforms=None):
        """
        Initialize with mean, std for normalization, k for the number of transformed images,
        image_size for resizing every image, and optionally a list of specific transformations.
        """
        self.mean = mean
        self.std = std
        self.k = k
        self.image_size = image_size
        if self.image_size is not None:
            self.resize_transform = T.Resize(self.image_size)
        else:
            self.resize_transform = None
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

        self.all_transforms = {
            # 'color_jitter': T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            # 'random_horizontal_flip': T.RandomHorizontalFlip(),
            # 'random_vertical_flip': T.RandomVerticalFlip(),
            # 'random_rotation': T.RandomRotation(30),
            # 'random_affine': T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            # 'grayscale': T.Grayscale(num_output_channels=3),
            # 'random_perspective': T.RandomPerspective(distortion_scale=0.5, p=0.5),
            # 'random_erasing': T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
            # 'gaussian_blur': T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # 'pad': T.Pad(padding=10, fill=(255, 0, 0), padding_mode='constant'),
            # # v2 Transforms
            # 'auto_augment': T.v2.AutoAugment(),
            # 'rand_augment': T.v2.RandAugment(),
            'aug_mix': T.v2.AugMix(),
            # # 'cut_mix': T.v2.CutMix(),
            # # 'mix_up': T.v2.MixUp(),
            # 'random_adjust_sharpness': T.v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            # 'random_autocontrast': T.v2.RandomAutocontrast(p=0.5),
            # 'random_equalize': T.v2.RandomEqualize(p=0.5),
            # 'random_invert': T.v2.RandomInvert(p=0.5),
            # 'random_posterize': T.v2.RandomPosterize(bits=4, p=0.5),
            # 'random_solarize': T.v2.RandomSolarize(threshold=128, p=0.5),
            # 'random_grayscale': T.v2.RandomGrayscale(p=0.1),
            # Add more transformations as needed
        }

        self.specific_transforms = specific_transforms if specific_transforms else self.all_transforms

    def __call__(self, image):
        """
        Apply resize and a different single transformation to the image for each of k images,
        and return these along with the resized original image.
        """
        resized_image = self.resize_transform(image)
        transformed_images = [self.to_tensor(resized_image)]  # Include the resized original image

        for i in range(self.k):
            
            # Randomly choose 1 transformation from the list of specific transformations
            transform_name, transform = random.choice(list(self.specific_transforms.items()))
            transformed_image = transform(resized_image.copy())
            transformed_image_tensor = self.to_tensor(transformed_image)
            normalized_image = self.normalize(transformed_image_tensor)
            transformed_images.append(normalized_image)

        return transformed_images
    
class CutMix:
    def __init__(self, alpha=1.0, num_classes=None):
        """
        Initialize the CutMix object with alpha and number of classes.

        Args:
            alpha (float): Hyperparameter for beta distribution.
            num_classes (int): Number of classes for creating soft labels.
        """
        if num_classes is None:
            raise ValueError("num_classes must be specified for label mixing.")
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, features, labels):
        """
        Apply CutMix to a batch of images and labels, modifying the labels with lambda.

        Args:
            features (Tensor): A batch of images of shape (B, C, H, W).
            labels (Tensor): A batch of labels of shape (B,).

        Returns:
            mixed_features (Tensor): The batch of CutMix images.
            mixed_labels (Tensor): The batch of mixed soft labels.
        """
        # The size of the batch
        B, C, H, W = features.size()
        device = features.device
        indices = torch.randperm(B).to(device)

        # Sample lambda from the beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # The bounding box for CutMix
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample the center of the bounding box
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Calculate the bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Create the mixed features
        mixed_features = features.clone()
        mixed_features[:, :, bby1:bby2, bbx1:bbx2] = features[indices, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Convert labels to one-hot format
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        labels_one_hot_perm = labels_one_hot[indices]

        # Mix the labels
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot_perm

        return mixed_features, mixed_labels

if __name__ == "__main__":

    # import clip
    # # Load the CLIP model
    # clip_model, clip_transform = clip.load('ViT-B/32', device='cpu')

    # # Extracting the values
    # normalize_values = next((t.mean, t.std) for t in clip_transform.transforms if isinstance(t, Normalize))
    # resize_value = next((t.size) for t in clip_transform.transforms if isinstance(t, Resize))

    # print(normalize_values, resize_value)

    # Define batch size, channels, height, and width for synthetic data
    batch_size, channels, height, width = 10, 3, 32, 32

    # Generate synthetic feature maps and labels
    features = torch.rand(batch_size, channels, height, width)
    labels = torch.randint(0, 10, (batch_size,))  # Assuming 10 classes

    cutmix = CutMix(alpha=1.0, num_classes=10)

    # Function call for CutMix
    mixed_features, mixed_labels = cutmix(features, labels)

    # Check the results
    print("Original features shape:", features.shape)
    print("Mixed features shape:", mixed_features.shape)
    print("Original labels:", labels)
    print("Mixed labels (labels, labels[indices], lambda):", mixed_labels)

    assert False
    # Get the mean and std from the clip_transform
    mean=[0.48145466, 0.4578275, 0.40821073]
    std=[0.26862954, 0.26130258, 0.27577711]
    image_size = (224, 224)

    k = 3  # Number of different transformed images to generate

    # Create an ImageTransformer instance
    clip_transform = ImageTransforms(mean, std, k, image_size)

    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_CIFAR10_dataloader(data_dir='./data',    
                                                            train_transform=None, test_transform=None, clip_transform=clip_transform,
                                                            subsample_trainset=False, return_dataset=True)

    # Load dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # Get a batch of images and labels
    images, labels, clip_batch = next(iter(train_loader))

    print(images.shape, labels.shape)

    # concatenate the clip_batch
    clip_batch = torch.cat(clip_batch, dim=0)

    print(clip_batch.shape)
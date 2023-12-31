import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import umap

import numpy as np
import os
from PIL import Image



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
    for i, reduced_tensor in enumerate(reduced_tensors):
        ax.scatter(reduced_tensor[:, 0], reduced_tensor[:, 1], color=colors[i], label=labels[i])

    # Draw lines between corresponding points for the first two tensors
    for i in range(len(tensor1_np)):
        points = np.vstack((reduced_tensors[0][i], reduced_tensors[1][i]))
        ax.plot(points[:, 0], points[:, 1], 'grey', alpha=0.5)

    # Optionally draw lines for the third tensor
    if tensor3 is not None and include_lines_for_tensor3 and len(tensor1_np) == len(tensor3_np):
        for i in range(len(tensor1_np)):
            points = np.vstack((reduced_tensors[0][i], reduced_tensors[2][i]))
            ax.plot(points[:, 0], points[:, 1], 'purple', alpha=0.5)

    # Customize the plot with legends
    ax.legend()
    ax.set_title('UMAP projection of the tensor embeddings', fontsize=18)

    plt.show()

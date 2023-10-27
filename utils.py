import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
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

def compute_similarities(image_embeddings, text_embeddings, mode='cosine'):
    if mode == 'cosine':
        return cosine_similarities(image_embeddings, text_embeddings)
    # TODO: add mean for DN
    elif mode == 'DN':
        return CLIP_DN_similarities(image_embeddings, text_embeddings)
    elif mode == 'DN*':
        cos_sim = cosine_similarities(image_embeddings, text_embeddings)
        dn_sim = CLIP_DN_similarities(image_embeddings, text_embeddings)
        return (cos_sim + dn_sim)/2

def cosine_similarities(image_embeddings, text_embeddings):
    """ Compute cosine similarities between image embeddings and text encodings for all labels """
    # image_embeddings = F.normalize(image_embeddings, dim=-1)
    # text_embeddings = F.normalize(text_embeddings, dim=-1)

    # make the text embeddings to the same data type as image embeddings
    text_embeddings = text_embeddings.type_as(image_embeddings)
    # similarities = text_embeddings.T @ image_embeddings
    similarities = F.cosine_similarity(image_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=2)
    
    return similarities

def CLIP_DN_similarities(image_embeddings, text_embeddings, image_embeddings_mean, text_embeddings_mean):
    "Compute cos similarity with distribution normalization"

    DN_image_embeddings = image_embeddings - image_embeddings_mean/2
    DN_text_embeddings = text_embeddings - text_embeddings_mean/2

    # Normalize the embeddings
    DN_image_embeddings = F.normalize(DN_image_embeddings, dim=-1)
    DN_text_embeddings = F.normalize(DN_text_embeddings, dim=-1)
    # Compute the dot product between the two tensors
    similarities = DN_image_embeddings.T @ DN_text_embeddings
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
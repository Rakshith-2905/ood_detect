import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import clip

from functools import reduce
from operator import mul
import numpy as np
from PIL import Image
import math

class VisualTransformer(nn.Module):
    def __init__(self, clip_model, input_dim=512, token_dim=768, num_positions=49):  # assuming num_positions to match 7x7 grid from conv1
        super(VisualTransformer, self).__init__()

        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        self.num_positions = num_positions

        # Linear layer to project your feature vector directly to token space
        self.feature_to_token = nn.Linear(input_dim, token_dim * num_positions)

    def embed_image(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x

    def embed_feature(self, feature_vector):
        B = feature_vector.shape[0]

        # Convert the feature vector to token representation
        tokens = self.feature_to_token(feature_vector) # shape = [*, token_dim * num_positions]
        tokens = tokens.view(B, self.num_positions, -1) # shape = [*, num_positions, token_dim]

        # Add class token, its of template [cls, emb, emb, emb, ...]
        tokens = torch.cat([self.visual.class_embedding.to(tokens.dtype) + torch.zeros(B, 1, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device), tokens], dim=1) # shape = [*, num_positions + 1, token_dim]

        # Add positional embedding
        tokens = tokens + self.visual.positional_embedding.to(tokens.dtype)

        return tokens

    def encoder(self, x):   
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x

    def freeze_all(self):
        """
        Freezes all the parameters of the model.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_specific(self, layer_name):
        """
        Unfreezes the specific layer provided by the layer_name.
        """
        if hasattr(self, layer_name):
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Layer {layer_name} not found in the model.")

    def set_trainable_layers(self, layers_list):
        """
        Given a list of layer names, sets them as trainable and the rest as frozen.
        """
        self.freeze_all()
        for layer in layers_list:
            self.unfreeze_specific(layer)
            
    def forward(self, x):
        # If the input is an image (detect based on the number of dimensions)
        if x.dim() == 4:  # NCHW format
            tokens = self.embed_image(x.type(self.dtype))
        else:  # If the input is a feature vector
            tokens = self.embed_feature(x)
            tokens = tokens.type(self.dtype)
        
        encoded = self.encoder(tokens)
        return encoded

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

        # Make it a 4-layer MLP
        # linear = nn.sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, output_dim),
        # )

    def forward(self, x):
        x = self.linear(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)    
    clip_model.eval()

    CLIP_image_encoder = VisualTransformer(clip_model, input_dim=512, token_dim=768, num_positions=49).to(device)

    # Load class names from a text file
    with open('data/domainnet_v1.0/class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    text_embedding = torch.load('prompts/text_embeddings.pth')[0]
    # Random feature vector
    feature_vector = torch.rand(1, 512).to(device)
    # Load and preprocess the image
    image = preprocess(Image.open("data/domainnet_v1.0/real/aircraft_carrier/real_001_000001.jpg")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = CLIP_image_encoder(image)
        feature_embedding = CLIP_image_encoder(feature_vector)
        
        orig_image_embedding = clip_model.encode_image(image)

    # Compute cosine similarity
    image_similarity = F.cosine_similarity(image_embedding, text_embedding, dim=-1)
    orig_image_similarity = F.cosine_similarity(orig_image_embedding, text_embedding, dim=-1)

    # Get the top 5 labels
    image_top5 = torch.topk(image_similarity, 5).indices
    orig_image_top5 = torch.topk(orig_image_similarity, 5).indices

    # Print the labels by indexing into the class names
    top5_names = [class_names[i] for i in image_top5]
    orig_top5_names = [class_names[i] for i in orig_image_top5]

    print(f"Image top 5: {top5_names}")
    print(f"Orig image top 5: {orig_top5_names}")
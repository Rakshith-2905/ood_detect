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
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual # visual backbone from CLIP
        self.dtype = clip_model.dtype 

    def embed_image(self, x):
        """
        The input is tensor image
        """
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x

    def encoder(self, x):
        """
        The input is after positional embedding
        """
    
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj
        return x

    def forward(self, image_input):
        x = self.embed_image(image_input.type(self.dtype))
        x = self.encoder(x)
        return x

class PromptedCLIPImageEncoder(nn.Module):
    def __init__(self, clip_model, prompt_initiation='random', 
                 dropout=0.0, num_tokens=16, prompt_dim=768, device='cpu'):
        super().__init__()

        for param in clip_model.parameters():
            param.requires_grad = False

        self.VisualTransformer = VisualTransformer(clip_model)

        patch_size = _pair((16, 16))

        num_tokens = num_tokens
        self.num_tokens = num_tokens  # number of prompted tokens

        self.prompt_dropout = Dropout(dropout)

        prompt_dim = 768

        self.dtype = clip_model.dtype
        # initiate prompt:
        if prompt_initiation == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim, dtype=self.dtype), requires_grad=True)  # (1, n_prompt, prompt_dim)
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            print("Prompt embeddings are randomly initialized")
        else:
            raise ValueError("Other initiation scheme is not supported")
        
    def incorporate_prompt(self, x):
        """
        The input is tensor image
        """
        B = x.shape[0]
        x = self.VisualTransformer.embed_image(x) # (batch_size, cls_token + n_patches, hidden_dim) output after positional embedding

        # incorporate prompt into the input features after cls token and before patches
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_embeddings).expand(B, -1, -1),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def forward(self, image_input):
        
        # Incorporate the prompt into the image features after the positional embedding
        embedding_output = self.incorporate_prompt(image_input.type(self.dtype))

        # Encode the image features
        encoded = self.VisualTransformer.encoder(embedding_output.type(self.dtype))

        return encoded

class PromptedCLIPTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx=16, num_classes=345, device='cpu', is_dist_prompt=False):
        super().__init__()
        
        self.clip_model = clip_model
        self.device = device
        self.is_dist_prompt = is_dist_prompt

        for param in self.clip_model.parameters():
            param.requires_grad = False


        dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        
        ctx_init = " ".join(["X"] * n_ctx)
        
        # use given words to initialize context vectors
        prompt = clip.tokenize(ctx_init).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        if self.is_dist_prompt:
            self.ctx = nn.Parameter(torch.randn_like(ctx_vectors, dtype=dtype))
        else:
            self.ctx = nn.ParameterList([nn.Parameter(torch.randn_like(ctx_vectors, dtype=dtype)) for i in range(num_classes)])

        self.dtype = dtype
        self.n_ctx = n_ctx

        # No gradients for the clip model parameters
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def compute_prefix_sufix(self, phrases):
        
        prompt_dummy = " ".join(["X"] * self.n_ctx)

        phrases = [phrase.replace("_", " ") for phrase in phrases]
        prompts = [prompt_dummy + " " + name for name in phrases]
        
        # Tokenize the prompt with the dummy preffix added
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)

        # Embed the tokens
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        # Split the prefix and suffix from the embeddings
        # prefix is start of sentence[SOS]: suffix is the actual phrase with EOS
        token_prefix = embedding[:, :1, :]  # (batch, 1, dim)
        token_suffix = embedding[:, 1 + self.n_ctx :, :] # (batch, *, dim)

        return token_prefix, token_suffix

    def forward(self, phrases):
        # Compute the prefix (SOS) and suffix (EOS) tokens for the phrases
        prefix, suffix = self.compute_prefix_sufix(phrases)

        if not self.is_dist_prompt:
            prompted_phrases = []
            for i in range(len(self.ctx)):
                # Concatenate the prefix, context, and suffix to form the new prompt
                prompts = torch.cat(
                    [
                        prefix[i],  # (batch, 1, dim)
                        self.ctx[i],     # (batch, n_ctx, ctx_dim)
                        suffix[i],  # (batch, *, dim)
                    ],
                    dim=0,
                )
                prompted_phrases.append(prompts)
            
            # Concatenate the prompted phrases
            prompted_phrases = torch.stack(prompted_phrases, dim=0)
        else:
            B = prefix.shape[0]
            # Concatenate the prefix, context, and suffix to form the new prompt
            prompted_phrases = torch.cat(
                [
                    prefix,  # (batch, 1, dim)
                    self.ctx.expand(B, -1, -1),     # (batch, n_ctx, ctx_dim)
                    suffix,  # (batch, *, dim)
                ],
                dim=1,
            )
        # Compute the embeddings for the prompted phrases
        text_encodings = self.encode_text(prompted_phrases, self.tokenized_prompts)
        
        return text_encodings

    def encode_text(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)    
    clip_model.eval()

    CLIP_image_encoder = VisualTransformer(clip_model)

    visual_prompter = PromptedCLIPImageEncoder(clip_model, num_tokens=8, device=device).to(device)
    text_prompter = PromptedCLIPTextEncoder(clip_model, num_classes=2, is_dist_prompt=False, device=device).to(device)
    
    # Load and preprocess the image
    image = preprocess(Image.open("models/donkey.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of a donkey", "a photo of a horse"]).to(device)
    with torch.no_grad():
        image_features = CLIP_image_encoder(image)

        prompted_image_features = visual_prompter(image)
        prompted_text_features = text_prompter(["a photo of a donkey", "a photo of a horse"])

        # Compare the text and image features
        logits_per_image, logits_per_text = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)  
    # Cos similarity between the image and text features
    print("Cosine similarity:", F.cosine_similarity(image_features, prompted_image_features).cpu().numpy())
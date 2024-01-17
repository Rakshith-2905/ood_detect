import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

from models.projector import ProjectionHead
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder

class PLUMBER(nn.Module):
    def __init__(self, clip_name, num_classes, img_projection=False, txt_projection=False, 
                 img_prompting=False, cls_txt_prompts=False, dataset_txt_prompt=False, 
                 is_mlp=False, device='cpu'):
        super().__init__()
        """
        Args:
            clip_model: CLIP model
            class_names: List of class names
            img_projection: Whether to use projection head for image embeddings
            txt_projection: Whether to use projection head for text embeddings
            img_prompting: Whether to use image prompting
            cls_txt_prompts: Whether to use class specific text prompts
            dataset_txt_prompt: Whether to use dataset level text prompts
            is_mlp: Whether to use MLP projection head or not
        """
        
        clip_model, preprocess = clip.load(clip_name, device=device)
        clip_model = clip_model.eval()

        # Set required grad for the clip model to False
        for param in clip_model.parameters():
            param.requires_grad = False

        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenize = clip.tokenize
        self.temperature = clip_model.logit_scale.exp()

        self.device = device

        print("\n\n Constructing PLUMBER \n")

        projection_dim = clip_model.visual.output_dim
        # Initialize image projector
        self.img_projector = None
        if img_projection:
            self.img_projector = ProjectionHead(input_dim=projection_dim, output_dim=projection_dim, is_mlp=is_mlp)
            self.img_projector = self.img_projector.to(device)
            print(f"Constructed image emb projection with is_mlp: {is_mlp}")

        # Initialize text projector
        self.text_projector = None
        if txt_projection:
            self.text_projector = ProjectionHead(input_dim=projection_dim, output_dim=projection_dim, is_mlp=is_mlp)
            self.text_projector = self.text_projector.to(device)
            print(f"Constructed text emb projection with is_mlp: {is_mlp}")

        # Initialize text encoder
        self.clip_prompted_txt_enc = None
        if cls_txt_prompts or dataset_txt_prompt:
            self.clip_prompted_txt_enc = PromptedCLIPTextEncoder(clip_model, num_classes=num_classes, 
                                                                 is_dist_prompt=dataset_txt_prompt, device=device)
            print(f"Constructed CLIP {'Class' if cls_txt_prompts else 'Dataset'} specific Prompted Text Encoder with {num_classes} classes")

        # Initialize image encoder
        self.clip_prompted_img_enc = None
        if img_prompting:
            self.clip_prompted_img_enc = PromptedCLIPImageEncoder(clip_model, device=device)
            print(f"Constructed CLIP Prompted Image Encoder")

        print("\nPLUMBER Constructed \n")

    def un_normalize(self, images):
        for t in self.preprocess.transforms:
            if isinstance(t, transforms.Normalize):
                mean = t.mean
                std = t.std
                break
        images = images * std[:, None, None] + mean[:, None, None]
        return images

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint
        Args:
            checkpoint_path: Path to checkpoint
        """

        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"\n\n")
        if self.img_projector:
            self.img_projector.load_state_dict(checkpoint['img_projector'])
            print(f"Loaded image projector from checkpoint: {checkpoint_path}")
        if self.text_projector:
            self.text_projector.load_state_dict(checkpoint['text_projector'])
            print(f"Loaded text projector from checkpoint: {checkpoint_path}")
        if self.clip_prompted_txt_enc:
            self.clip_prompted_txt_enc.load_state_dict(checkpoint['clip_prompted_txt_enc'])
            print(f"Loaded CLIP Prompted Text Encoder from checkpoint: {checkpoint_path}")
        if self.clip_prompted_img_enc:
            self.clip_prompted_img_enc.load_state_dict(checkpoint['clip_prompted_img_enc'])
            print(f"Loaded CLIP Prompted Image Encoder from checkpoint: {checkpoint_path}")

        print("\nCheckpoint Loaded \n")

    def reset(self):
        """
        Reset model parameters
        """
        print("\n\n Resetting model parameters \n")
        self.load_checkpoint(self.checkpoint_path)

        print("\nModel parameters reset \n")

    def save_checkpoint(self, checkpoint_path):
        """
        Save checkpoint
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = {}

        if self.img_projector:
            checkpoint['img_projector'] = self.img_projector.state_dict()
        if self.text_projector:
            checkpoint['text_projector'] = self.text_projector.state_dict()
        if self.clip_prompted_txt_enc:
            checkpoint['clip_prompted_txt_enc'] = self.clip_prompted_txt_enc.state_dict()
        if self.clip_prompted_img_enc:
            checkpoint['clip_prompted_img_enc'] = self.clip_prompted_img_enc.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    def optimizer_init(self, optimizer_name, learning_rate):
        """
        Set optimizer for the model
        Args:
            optimizer_name: Name of the optimizer
            learning_rate: Learning rate for the optimizer
        """
        print("\n\n Initializing Optimizer(s) \n")
        
        self.optimizer_img_proj = None
        if self.img_projector:
            if optimizer_name == 'adam':
                self.optimizer_img_proj = torch.optim.Adam(self.img_projector.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                self.optimizer_img_proj = torch.optim.SGD(self.img_projector.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer_name == 'adamw':
                self.optimizer_img_proj = torch.optim.AdamW(self.img_projector.parameters(), lr=learning_rate)

            print(f"Constructed optimizer for image projector: {optimizer_name} with lr: {learning_rate}")
        
        self.optimizer_txt_proj = None
        if self.text_projector:
            if optimizer_name == 'adam':
                self.optimizer_txt_proj = torch.optim.Adam(self.text_projector.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                self.optimizer_txt_proj = torch.optim.SGD(self.text_projector.parameters(), lr=learning_rate, momentum=0.9)
            elif optimizer_name == 'adamw':
                self.optimizer_txt_proj = torch.optim.AdamW(self.text_projector.parameters(), lr=learning_rate)

            print(f"Constructed optimizer for text projector: {optimizer_name} with lr: {learning_rate}")
        
        self.optimizer_txt_prompt = None
        if self.clip_prompted_txt_enc:
            self.optimizer_txt_prompt = torch.optim.SGD([p for p in self.clip_prompted_txt_enc.parameters() if p.requires_grad], lr=0.1)
            print(f"Constructed optimizer for text prompt: SGD with lr: 0.1")

        self.optimizer_img_prompt = None
        if self.clip_prompted_img_enc:
            self.optimizer_img_prompt = torch.optim.SGD([p for p in self.clip_prompted_img_enc.parameters() if p.requires_grad], lr=0.1)
            print(f"Constructed optimizer for image prompt: SGD with lr: 0.1")

        self.optimizers_dict = {
            "optimizer_img_proj": self.optimizer_img_proj,
            "optimizer_txt_proj": self.optimizer_txt_proj,
            "optimizer_txt_prompt": self.optimizer_txt_prompt,
            "optimizer_img_prompt": self.optimizer_img_prompt
        }

        print("\nOptimizer(s) Initialized \n")

        return self.optimizers_dict
    
    def scheduler_init(self, milestones=[30, 60, 90], gamma=0.1):
        """
        Set scheduler for the model
        Args:
            scheduler_name: Name of the scheduler
            milestones: List of milestones for the scheduler
            gamma: Gamma for the scheduler
        """
        print("\n\n Initializing Scheduler(s) \n")
        scheduler_img_proj = None
        if self.img_projector:
            scheduler_img_proj = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_img_proj, milestones=milestones, gamma=gamma)
        scheduler_txt_proj = None
        if self.text_projector:
            scheduler_txt_proj = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_txt_proj, milestones=milestones, gamma=gamma)

        self.schedulers_dict = {
            "scheduler_img_proj": scheduler_img_proj,
            "scheduler_txt_proj": scheduler_txt_proj
        }

        print("\nScheduler(s) Initialized \n")

        return self.schedulers_dict

    def optimizer_step(self):
        """
        Perform optimizer step
        """
        for optimizer_name, optimizer in self.optimizers_dict.items():
            if optimizer:
                optimizer.step()

    def zero_grad(self):
        """
        Zero grad for all optimizers
        """
        for optimizer_name, optimizer in self.optimizers_dict.items():
            if optimizer:
                optimizer.zero_grad()

    def scheduler_step(self):
        """
        Perform scheduler step
        """
        for scheduler_name, scheduler in self.schedulers_dict.items():
            if scheduler:
                scheduler.step()

    def set_train_mode(self):
        if self.img_projector:
            self.img_projector.train()
        if self.text_projector:
            self.text_projector.train()
        if self.clip_prompted_txt_enc:
            self.clip_prompted_txt_enc.train()
        if self.clip_prompted_img_enc:
            self.clip_prompted_img_enc.train()

    def set_eval_mode(self):
        if self.img_projector:
            self.img_projector.eval()
        if self.text_projector:
            self.text_projector.eval()
        if self.clip_prompted_txt_enc:
            self.clip_prompted_txt_enc.eval()
        if self.clip_prompted_img_enc:
            self.clip_prompted_img_enc.eval()
        
    def encode_images(self, images):
        """
        Encode images using CLIP model
        Args:
            images: Tensor of shape [batch_size, 3, H, W], or list of PIL images
        Returns:
            proj_embeddings: Tensor of shape [batch_size, proj_dim]
        """

        # If images is a list of PIL images, convert to tensor using preprocess
        if isinstance(images, list):
            images = torch.stack([self.preprocess(image) for image in images])

        # Image encoding logic
        if self.clip_prompted_img_enc:
            clip_image_embeddings = self.clip_prompted_img_enc(images)
        else:
            clip_image_embeddings = self.clip_model.encode_image(images)
        clip_image_embeddings = clip_image_embeddings.float()

        if self.img_projector:
            proj_embeddings = self.img_projector(clip_image_embeddings)
        else:
            proj_embeddings = clip_image_embeddings
        
        return proj_embeddings

    def encode_text(self, text_list, text_encodings_raw=None):
        """
        Encode text using CLIP model
        Args:
            text_list: List of text strings
        Returns:
            text_encodings: Tensor of shape [batch_size, proj_dim]
        """
        # If clip prompted text encoder is present, use it else return the text encodings raw
        if self.clip_prompted_txt_enc:
            text_encodings_raw = self.clip_prompted_txt_enc(text_list)
        elif text_encodings_raw is None:
            # Tokenize text
            text_list = self.tokenize(text_list).to(self.device)
            text_encodings_raw = self.clip_model.encode_text(text_list)
        text_encodings_raw = text_encodings_raw.float()
        # If text projector is present, project the text embeddings
        if self.text_projector:
            text_encodings = self.text_projector(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw

        return text_encodings
    
    def forward(self, images, text_list):
        """
        Performs classification using modified CLIP model
        Args:
            images: Tensor of shape [batch_size, 3, H, W] or list of PIL images
            text_list: List of text strings
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        # Encode images and text
        image_embeddings = self.encode_images(images)
        text_embeddings = self.encode_text(text_list)

        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute logits
        logits = self.temperature*image_embeddings @ text_embeddings.t()
        
        return logits


if __name__ == "__main__":

    plumber = PLUMBER(clip_name='ViT-B/32', num_classes=10, img_projection=True, txt_projection=True,
                        img_prompting=True, cls_txt_prompts=True, dataset_txt_prompt=True, is_mlp=True)
    print(plumber)
    
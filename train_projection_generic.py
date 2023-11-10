import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

import lightning as L
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger

import argparse
import os
from tqdm import tqdm
from functools import partial
from datetime import datetime

import clip
from YFCC_feature_extract import ImageTextDataset
from models.ViT_models import SAMBackbone, MAEBackbone, DINOBackbone
from models.resnet import CustomFeatureModel
from models.projector import ProjectionHead
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow

def get_save_dir(args):
    # Add time stamp to the save directory

    save_dir = os.path.join(args.save_dir, args.feature_extractor_name)
    save_dir += f"{args.prefix}"
    save_dir += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return save_dir

def progbar_wrapper(iterable, total, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if fabric.is_global_zero:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
    
def train_one_epoch(train_loader, clip_model, feature_extractor, projector, criterion, optimizer, epoch):
    clip_model.eval()
    feature_extractor.eval()
    projector.train()
    
    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    pbar = progbar_wrapper(
        train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}"
    )
    for images_batch, images_clip_batch, captions_batch, image_names_batch in pbar:

        optimizer.zero_grad()
        
        # clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Extract features for images and text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions_batch)
            clip_txt_embeddings = clip_model.encode_text(text_tokens).detach().cpu()

        custom_image_embeddings = feature_extractor(images_batch)

        # Project the resnet embeddings
        proj_embeddings = projector(custom_image_embeddings)

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(clip_txt_embeddings, dim=-1)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # The logits dimension (batch_size, batch_size)
        logits_per_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # 100 is the logits scale from CLIP
        logits_per_text = logits_per_projection.t()

        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries

        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(proj_embeddings)).long() # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2
        
        fabric.backward(loss)

        optimizer.step()

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()
        
        pbar.set_postfix({"Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})

    # TODO: Check if this is correct
    total_loss = fabric.all_gather(total_loss).sum() / len(train_loader)
    total_image_loss = fabric.all_gather(total_image_loss).sum() / len(train_loader)
    total_text_loss = fabric.all_gather(total_text_loss).sum() / len(train_loader)

    return  total_loss, total_image_loss, total_text_loss

@torch.no_grad()
def validate(val_loader, clip_model, feature_extractor, projector, criterion, epoch, label_mapping=None):
    
    clip_model.eval()
    feature_extractor.eval()
    projector.train()
    
    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    pbar = progbar_wrapper(
        val_loader, total=len(val_loader), desc=f"Validation Epoch {epoch+1}"
    )
    for images_batch, images_clip_batch, captions_batch, image_names_batch in pbar:

        clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        text_tokens = clip.tokenize(captions_batch)
        clip_txt_embeddings = clip_model.encode_text(text_tokens).detach().cpu()

        custom_image_embeddings = feature_extractor(images_batch)

        # Project the resnet embeddings
        proj_embeddings = projector(custom_image_embeddings)

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(clip_txt_embeddings, dim=-1)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # The logits dimension (batch_size, batch_size)
        logits_per_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # 100 is the logits scale from CLIP
        logits_per_text = logits_per_projection.t()

        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries

        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(proj_embeddings)).long() # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()
        
        pbar.set_postfix({"Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})

    # TODO: Check if this is correct
    # all_gather is used to aggregated the value across processes
    total_loss = fabric.all_gather(total_loss).sum() / len(val_loader)
    total_image_loss = fabric.all_gather(total_image_loss).sum() / len(val_loader)
    total_text_loss = fabric.all_gather(total_text_loss).sum() / len(val_loader)

    return  total_loss, total_image_loss, total_text_loss

def build_feature_extractor(feature_extractor_name, feature_extractor_checkpoint_path=None):
    """
    Builds the feature extractor based on the provided name.
    Args:
        feature_extractor_name (str): The name of the feature extractor to use.
    Returns:
        torch.nn.Module: The feature extractor.
    """
    if args.feature_extractor_name == 'sam_vit_h':
        if feature_extractor_checkpoint_path is None:
            feature_extractor_checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"
        feature_extractor = SAMBackbone("vit_h", feature_extractor_checkpoint_path)
    elif args.feature_extractor_name == 'mae_vit_large_patch16':
        if feature_extractor_checkpoint_path is None:
            feature_extractor_checkpoint_path = "checkpoints/mae_visualize_vit_large_ganloss.pth"
        feature_extractor = MAEBackbone("mae_vit_large_patch16", feature_extractor_checkpoint_path)
    elif args.feature_extractor_name == 'dino_vits16':
        feature_extractor = DINOBackbone("dino_vits16", None)
    elif args.feature_extractor_name in ['resnet18', 'resnet50', 'resnet101', 'resnet50x1_bitm', 'resnetv2_101x1_bit.goog_in21k']:
        feature_extractor = CustomFeatureModel(args.feature_extractor_name, use_pretrained=True)
    else:
        raise NotImplementedError(f"{feature_extractor_name} is not implemented.")

    transform = feature_extractor.transform
    return feature_extractor, transform

def main(args):
    
    # Load the CLIP model
    clip_model, clip_preprocess = clip.load(args.clip_model_name)

    feature_extractor, transform = build_feature_extractor(args.feature_extractor_name)

    projector = ProjectionHead(input_dim=feature_extractor.feature_dim, output_dim=args.projection_dim)

    # Create the data loader and wrap them with Fabric
    train_dataset = ImageTextDataset(args.data_dir, args.json_file, start_index=args.train_start_index, end_index=args.train_end_index, 
                                        transform=transform, transform2=clip_preprocess)
    val_dataset = ImageTextDataset(args.data_dir, args.json_file, start_index=args.val_start_index, end_index=args.val_end_index, 
                                        transform=transform, transform2=clip_preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # Create the optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(projector.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(projector.parameters(), lr=args.learning_rate, momentum=0.9)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)


    clip_model = fabric.to_device(clip_model)
    feature_extractor = fabric.to_device(feature_extractor)

    # Wrap the feature extractor and optimizer with Fabric
    projector, optimizer = fabric.setup(projector, optimizer)

    # Make directory for saving results
    save_dir = get_save_dir(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    fabric.print(f"Results will be saved to {save_dir}")
    

    if fabric.is_global_zero:
        # Save arguments
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        fabric.barrier()
    
    start_epoch = 0
    state = {"projector": projector, "optimizer": optimizer, "epoch": start_epoch}

    tb_logger = TensorBoardLogger(save_dir)
    csv_logger = CSVLogger(save_dir)
    fabric.loggers = [tb_logger, csv_logger]

    if args.resume_checkpoint_path:
        fabric.load(args.resume_checkpoint_path, state)
        start_epoch = state["epoch"] + 1

    if start_epoch >= args.num_epochs:
        fabric.print(f"Already finished training for {args.num_epochs} epochs. Exiting...")
        return

    val_loss = float("inf")
    val_image_loss = float("inf")
    val_text_loss = float("inf")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,  train_image_loss, train_text_loss = train_one_epoch(train_loader, clip_model, feature_extractor, projector, 
                                                                                    criterion, optimizer, epoch)

        if epoch % args.val_freq == 0:
            val_loss, val_image_loss, val_text_loss = validate(val_loader, clip_model, feature_extractor, projector, 
                                                                        criterion, epoch)

        fabric.print(f"{epoch}/{args.num_epochs}| Total Train Loss: {train_loss:.4f}, Train Image Loss: {train_image_loss:.4f}, Train Text Loss: {train_text_loss:.4f}, Total Val Loss: {val_loss:.4f}, Val Image Loss: {val_image_loss:.4f}, Val Text Loss: {val_text_loss:.4f}")

        losses_dict = {"train_loss": train_loss, "train_image_loss": train_image_loss, "train_text_loss": train_text_loss,
                        "val_loss": val_loss, "val_image_loss": val_image_loss, "val_text_loss": val_text_loss}

        fabric.log_metrics(losses_dict, step=epoch)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            state.update(epoch=epoch)
            fabric.save(os.path.join(save_dir, "best_projector_weights.pth"), state)
        
        if epoch % 10 == 0:
            state.update(epoch=epoch)
            fabric.save(os.path.join(save_dir, "projector_weights.pth"), state)

    fabric.save(os.path.join(save_dir, "projector_weights.pth"), state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='data/domainnet_v1.0', help='Path to the data directory')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file.')
    parser.add_argument('--train_start_index', type=int, default=0, help='The starting line index in the JSON file.')
    parser.add_argument('--train_end_index', type=int, help='The ending line index in the JSON file.')
    parser.add_argument('--val_start_index', type=int, default=0, help='The starting line index in the JSON file.')
    parser.add_argument('--val_end_index', type=int, help='The ending line index in the JSON file.')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--feature_extractor_name', required=True, choices=['sam_vit_h', 'mae_vit_large_patch16', 'dino_vits16', 'resnet50x1_bitm', 'resnetv2_101x1_bit.goog_in21k'],  help='Name of the feature extractor to use.')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--teacher_temp', type=float, default=0.5, help='Temperature for Dino loss')
    parser.add_argument('--student_temp', type=float, default=1, help='Temperature for Dino loss')

    parser.add_argument('--num_gpus', type=int, default=8, help='Number of gpus for DDP')

    args = parser.parse_args()

    # Print the arguments
    print(args)


    fabric = L.Fabric(accelerator="cuda", devices=args.num_gpus, strategy="ddp")
    fabric.launch()
    
    seed_everything(args.seed)

    main(args)

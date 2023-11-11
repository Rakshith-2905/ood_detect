import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import Accuracy
import utils_ddp

import argparse
import os
from datetime import datetime
import clip

from models.ViT_models import SAMBackbone, MAEBackbone, DINOBackbone
from models.resnet import CustomFeatureModel
from models.projector import ProjectionHead
from YFCC_feature_extract import ImageTextDataset
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow

# def setup(args):
#     args.world_size =int(os.environ['OMPI_COMM_WORLD_SIZE'])
#     args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#     args.gpu = 0#int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
#     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     dist.barrier()
#     setup_for_distributed(args.rank == 0)
    

def cleanup():
    dist.destroy_process_group()

def init_csv_logger(filename, fieldnames):
    """Initialize a CSV logger."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def log_to_csv(filename, data_dict):
    """Log data to a CSV file."""
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        writer.writerow(data_dict)

def get_save_dir(args):
    
    # If resume_checkpoint_path is provided, then use the save_dir from that checkpoint
    if args.resume_checkpoint_path:
        save_dir = os.path.dirname(args.resume_checkpoint_path)
        return save_dir

    save_dir = os.path.join(args.save_dir, args.feature_extractor_name)
    save_dir += f"{args.prefix}"
    # save_dir += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    return save_dir

def reduce_tensor(tensor, rank):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def progbar_wrapper(iterable, total, rank, **kwargs):
    """Wraps the iterable with tqdm for global rank zero.

    Args:
        iterable: the iterable to wrap with tqdm
        total: the total length of the iterable, necessary in case the number of batches was limited.

    """
    if rank == 0:
        return tqdm(iterable, total=total, **kwargs)
    return iterable
    
def train_one_epoch(train_loader, clip_model, feature_extractor, projector, criterion, optimizer, epoch, rank):
    clip_model.eval()
    feature_extractor.eval()
    projector.train()
    
    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    pbar = progbar_wrapper(train_loader, total=len(train_loader), rank=rank, desc=f"Training Epoch {epoch + 1}")
    for images_batch, images_clip_batch, captions_batch, image_names_batch in pbar:

        optimizer.zero_grad()
        
        # Ensure data is on the correct device
        images_batch = images_batch.to(rank)
        captions_batch = [caption for caption in captions_batch]

        # clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Extract features for images and text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions_batch, truncate=True).to(rank)
            clip_txt_embeddings = clip_model.encode_text(text_tokens)

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
        pseudo_labels = fabric.to_device(torch.arange(len(proj_embeddings)).long()) # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2
        
        loss.backward(loss)

        optimizer.step()

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        if rank == 0:
            pbar.set_description({"Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})

    # TODO: CHECK Reduce losses across all processes
    total_loss = reduce_tensor(total_loss, rank).item()/len(train_loader)
    total_image_loss = reduce_tensor(total_image_loss, rank).item()/len(train_loader)
    total_text_loss = reduce_tensor(total_text_loss, rank).item()/len(train_loader)

    return total_loss, total_image_loss, total_text_loss

@torch.no_grad()
def validate(val_loader, clip_model, feature_extractor, projector, criterion, epoch, rank):
    
    clip_model.eval()
    feature_extractor.eval()
    projector.train()
    
    total_loss = 0
    total_image_loss = 0
    total_text_loss = 0

    pbar = progbar_wrapper(train_loader, total=len(train_loader), rank=rank, desc=f"Validation Epoch {epoch + 1}")
    for images_batch, images_clip_batch, captions_batch, image_names_batch in pbar:

        # Ensure data is on the correct device
        images_batch = images_batch.to(rank)
        captions_batch = [caption for caption in captions_batch]

        # clip_image_embeddings = clip_model.encode_image(images_clip_batch)

        # Extract features for images and text
        with torch.no_grad():
            text_tokens = clip.tokenize(captions_batch, truncate=True).to(rank)
            clip_txt_embeddings = clip_model.encode_text(text_tokens)

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
        pseudo_labels = fabric.to_device(torch.arange(len(proj_embeddings)).long()) # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()

        if rank == 0:
            pbar.set_description({"Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})

    # TODO: CHECK Reduce losses across all processes
    total_loss = reduce_tensor(total_loss, rank).item()/len(val_loader)
    total_image_loss = reduce_tensor(total_image_loss, rank).item()/len(val_loader)
    total_text_loss = reduce_tensor(total_text_loss, rank).item()/len(val_loader)

    return total_loss, total_image_loss, total_text_loss


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

    try:
        utils_ddp.init_distributed_mode_lassen(args)
    except:
        assert Exception("Failed to initialize distributed mode")
    
    world_size = utils_ddp.get_world_size()
    rank = utils_ddp.get_rank()
    
    if rank == 0:
        # Make directory for saving results
        args.save_dir = get_save_dir(args)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        
        # Print the arguments
        print(f"Arguments: {args}")
        print(f"Results will be saved to {args.save_dir}")

        # Save arguments to a file
        with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

    # Load the CLIP model and build feature extractor, projector
    # Ensure models and data are on the correct device
    clip_model, _ = clip.load(args.clip_model_name, device=rank)
    feature_extractor, transform = build_feature_extractor(args.feature_extractor_name)
    feature_extractor.to(rank)
    feature_extractor = DDP(feature_extractor, device_ids=[rank])

    projector = ProjectionHead(input_dim=feature_extractor.module.feature_dim, output_dim=args.projection_dim)
    projector.to(rank)
    projector = DDP(projector, device_ids=[rank])

    # Create the optimizer
    optimizer = torch.optim.Adam(projector.parameters(), lr=args.learning_rate)

    # Create Distributed Samplers and DataLoaders
    train_dataset = ImageTextDataset(args.json_file, args.data_dir, start_index=args.train_start_index, end_index=args.train_end_index, 
                                        transform=transform, transform2=clip_preprocess)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    val_dataset = ImageTextDataset(args.json_file, args.data_dir, start_index=args.val_start_index, end_index=args.val_end_index, 
                                        transform=transform, transform2=clip_preprocess)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)


    # Create the optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(projector.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(projector.parameters(), lr=args.learning_rate, momentum=0.9)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Load checkpoint if available
    if args.resume_checkpoint_path and os.path.isfile(args.resume_checkpoint_path):
        checkpoint = torch.load(args.resume_checkpoint_path, map_location='cpu')
        projector.module.load_state_dict(checkpoint['projector_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    if rank == 0:
        csv_filename = os.path.join(args.save_dir, "training_log.csv")
        fieldnames = ['epoch', 'train_loss', 'train_image_loss', 'train_text_loss', 'val_loss', 'val_image_loss', 'val_text_loss']
        init_csv_logger(csv_filename, fieldnames)


    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    val_loss = float("inf")
    val_image_loss = float("inf")
    val_text_loss = float("inf")

    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_image_loss, train_text_loss = train_one_epoch(train_loader, clip_model, feature_extractor, projector, 
                                                                        criterion, optimizer, epoch, rank)

        if epoch % args.val_freq == 0:
            val_loss, val_image_loss, val_text_loss = validate(val_loader, clip_model, feature_extractor, projector, 
                                                            criterion, epoch, rank)

        if rank == 0:
            print(f"{epoch}/{args.num_epochs}| Total Train Loss: {train_loss:.4f}, Train Image Loss: {train_image_loss:.4f}, Train Text Loss: {train_text_loss:.4f}, Total Val Loss: {val_loss:.4f}, Val Image Loss: {val_image_loss:.4f}, Val Text Loss: {val_text_loss:.4f}")
            
            # Log data to CSV
            log_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_image_loss': train_image_loss,
                'train_text_loss': train_text_loss,
                'val_loss': val_loss,
                'val_image_loss': val_image_loss,
                'val_text_loss': val_text_loss
            }
            log_to_csv(csv_filename, log_data)

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'projector_state': projector.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, "best_projector_weights.pth"))
            
            if epoch % 10 == 0:
                checkpoint['epoch'] = epoch
                torch.save(checkpoint, os.path.join(args.save_dir, "projector_weights.pth"))

    if rank == 0:
        torch.save(checkpoint, os.path.join(args.save_dir, "projector_weights.pth"))

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

    parser.add_argument('--feature_extractor_name', required=True,  help='Name of the feature extractor to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k')
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

    main(args)
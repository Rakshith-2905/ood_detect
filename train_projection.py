import torch
import torch.nn.functional as F
import clip

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial


from models.resnet import CustomResNet
from models.visual_transformer import ProjectionHead, VisualTransformer
from domainnet_data import DomainNetDataset, get_domainnet_loaders, get_data_from_saved_files
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow
from prompts.FLM import generate_label_mapping_by_frequency, label_mapping_base

def get_save_dir(args):
    base_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    save_dir = os.path.join(base_dir, "projection")
    if args.use_default_prompt == True:
        save_dir += "_default_prompt"
    else:
        save_dir += "_FLM"
    if args.feature_similarity:
        save_dir += f"_feat_sim{args.feature_sim_weight}_distill{args.distill_loss_weight}"

    save_dir += f"_{args.similarity_mode}"
    save_dir += f"_mapping{args.mapping_num}"
    save_dir += "_scaled_logits_contra"

    return save_dir

def train_one_epoch(train_loader, resnet_model, projector, text_encodings, criterion, optimizer, device, epoch, label_mapping=None):
    resnet_model.eval()
    projector.train()
    
    total_loss = 0
    total_acc = 0
    total_image_loss = 0
    total_text_loss = 0

    total_samples = len(train_loader.dataset)

    # Wrap the train_loader with tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Training epoch: {epoch+1}')
    for resnet_logits, resnet_embeddings, gt_labels, CLIP_embeddings in pbar:

        resnet_logits, resnet_embeddings, gt_labels, CLIP_embeddings = resnet_logits.to(device), resnet_embeddings.to(device), gt_labels.to(device), CLIP_embeddings.to(device)
        
        optimizer.zero_grad()
        
        # Project the resnet embeddings
        proj_embeddings = projector(resnet_embeddings)

        # Create the text_encodings based on the gt labels
        gt_labels_list = gt_labels.cpu().numpy().tolist()
        gt_text_encodings = text_encodings[gt_labels_list] # (batch_size, CLIP_embedding_dim)

        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_gt_text_encodings = F.normalize(gt_text_encodings, dim=-1)

        # make the text embeddings to the same data type as image embeddings
        normalized_gt_text_encodings = normalized_gt_text_encodings.type_as(normalized_proj_embeddings)
        # The logits dimension (batch_size, batch_size)
        logits_per_projection = 100*normalized_proj_embeddings @ normalized_gt_text_encodings.t() # 100 is the logits scale from CLIP
        logits_per_text = logits_per_projection.t()

        # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries

        # labels are indexes to the diagonal entries of the logits matrix
        pseudo_labels = torch.arange(len(proj_embeddings)).long().to(device) # (batch_size)

        loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
        loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
        loss = (loss_image + loss_text)/2
        loss.backward()

        optimizer.step()

        # Probs from logits
        zero_shot_logits = compute_similarities(proj_embeddings, text_encodings, mode='cosine') # (batch_size, num_classes)
        projection_probs = F.softmax(zero_shot_logits, dim=-1)

        # Compute the accuracy
        batch_acc = compute_accuracy(projection_probs, gt_labels)

        total_acc += batch_acc

        batch_loss = loss.item() 
        total_loss += batch_loss
        
        total_image_loss += loss_image.item()
        total_text_loss += loss_text.item()
        
        # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_clip_acc})
        pbar.set_postfix({"Batch Acc":batch_acc,  "Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})
    return  total_acc/len(train_loader), total_loss/len(train_loader), total_image_loss/len(train_loader), total_text_loss/len(train_loader)


def validate(val_loader, resnet_model, projector, text_encodings, criterion, device, epoch, label_mapping=None):
    resnet_model.eval()
    projector.eval()
    
    total_loss = 0
    total_acc = 0
    total_image_loss = 0
    total_text_loss = 0

    total_samples = len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating epoch: {epoch+1}')
    with torch.no_grad():
        for resnet_logits, resnet_embeddings, gt_labels, CLIP_embeddings in pbar:

            resnet_logits, resnet_embeddings, gt_labels, CLIP_embeddings = resnet_logits.to(device), resnet_embeddings.to(device), gt_labels.to(device), CLIP_embeddings.to(device)                
            
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)

            # Create the text_encodings based on the gt labels
            gt_labels_list = gt_labels.cpu().numpy().tolist()
            gt_text_encodings = text_encodings[gt_labels_list] # (batch_size, CLIP_embedding_dim)

            normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
            normalized_gt_text_encodings = F.normalize(gt_text_encodings, dim=-1)

            # make the text embeddings to the same data type as image embeddings
            normalized_gt_text_encodings = normalized_gt_text_encodings.type_as(normalized_proj_embeddings)
            # The logits dimension (batch_size, batch_size)
            logits_per_projection = 100*normalized_proj_embeddings @ normalized_gt_text_encodings.t() # 100 is the logits scale from CLIP
            logits_per_text = logits_per_projection.t()

            # We want to maximize the diagonal entries of the logits matrix while minimizing the off-diagonal entries

            # labels are indexes to the diagonal entries of the logits matrix
            pseudo_labels = torch.arange(len(proj_embeddings)).long().to(device) # (batch_size)

            loss_image = F.cross_entropy(logits_per_projection, pseudo_labels)
            loss_text = F.cross_entropy(logits_per_text, pseudo_labels)
            loss = (loss_image + loss_text)/2

            # Probs from logits
            zero_shot_logits = compute_similarities(proj_embeddings, text_encodings, mode='cosine') # (batch_size, num_classes)
            projection_probs = F.softmax(zero_shot_logits, dim=-1)
            # Compute the accuracy
            batch_acc = compute_accuracy(projection_probs, gt_labels)

            batch_loss = loss.item() 
            total_loss += batch_loss
            total_acc += batch_acc
            
            total_image_loss += loss_image.item()
            total_text_loss += loss_text.item()
            
            # pbar.set_postfix({"Batch Loss": batch_loss, "Base model Acc": batch_base_model_acc, "CLIP Acc": batch_clip_acc})
            pbar.set_postfix({"Batch Acc":batch_acc,  "Batch Loss": batch_loss, "Image Loss": loss_image.item(), "Text Loss": loss_text.item()})
    return   total_acc/len(val_loader), total_loss/len(val_loader), total_image_loss/len(val_loader), total_text_loss/len(val_loader)

def main(args):

    base_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class names from a text file
    os.path.join(args.data_dir, 'class_names.txt')
    with open(os.path.join(args.data_dir, 'class_names.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # loaders, _ = get_domainnet_loaders(args.domain, args.batch_size)
    # val_loader = loaders['test']

    train_loader,val_loader = get_data_from_saved_files(os.path.join(base_dir, 'features'), args.batch_size, train_shuffle=True)

    # Load your trained model from checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    
    resnet_model = CustomResNet(model_name=args.resnet_model, num_classes=345)
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    resnet_model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    resnet_model.to(device)
    
    # # Load CLIP
    # clip_model, _ = clip.load('ViT-B/32', device=device)
    # projector = VisualTransformer(clip_model, input_dim=args.resnet_dim, token_dim=768, num_positions=49).to(device)
    # projector.set_trainable_layers(['feature_to_token'])

    projector = ProjectionHead(input_dim=args.resnet_dim, output_dim=args.projection_dim).to(device)

    projector.train()
    projector.requires_grad_(True)

    prompt_embeddings = torch.load(args.prompt_embeddings_pth)

    if args.use_default_prompt:
        text_encodings = prompt_embeddings[0]
    else:
        # Merge all the prompt embeddings into one tensor
        text_encodings = torch.cat(prompt_embeddings, dim=0)

    print(f"Loaded text encodings of shape: {text_encodings.shape}")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(projector.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(projector.parameters(), lr=args.learning_rate, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)


    # Make directory for saving results
    save_dir = get_save_dir(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving results to {save_dir}")
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Training and validation loop
    train_losses, val_losses = [], []
    train_image_losses, val_image_losses = [], []
    train_text_losses, val_text_losses = [], []

    train_accs, val_accs = [], []

    best_val_loss = 10000000000000
    for epoch in range(args.num_epochs):

        if epoch % args.mapping_interval == 0 and args.use_default_prompt == False:
            # Compute the text encoding using FLM
            mapping_sequence = generate_label_mapping_by_frequency(resnet_model, projector, compute_similarities, 
                                                                    train_loader, mapping_num = args.mapping_num, 
                                                                    similarity_mode=args.similarity_mode, text_encodings=text_encodings)

            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        else:
            label_mapping = None

        train_acc, train_loss,  train_image_loss, train_text_loss = train_one_epoch(train_loader, resnet_model, projector, text_encodings, criterion, optimizer, device, epoch, label_mapping=label_mapping)
        val_acc, val_loss, val_image_loss, val_text_loss = validate(val_loader, resnet_model, projector, text_encodings, criterion, device, epoch, label_mapping=label_mapping)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        train_image_losses.append(train_image_loss)
        val_image_losses.append(val_image_loss)
        train_text_losses.append(train_text_loss)
        val_text_losses.append(val_text_loss)

        print(f"Epoch {epoch+1}, Train Acc: {train_acc}, Total Train Loss: {train_loss:.4f}, Train Image Loss: {train_image_loss:.4f}, Train Text Loss: {train_text_loss:.4f}, Val Acc: {val_acc}, Total Val Loss: {val_loss:.4f}, Val Image Loss: {val_image_loss:.4f}, Val Text Loss: {val_text_loss:.4f}")

        # print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train feature loss: {train_feature_loss:.4f}, Train Base Model Accuracy: {train_base_acc:.4f}, Train CLIP Accuracy: {train_clip_acc:.4f}, Val Loss: {val_loss:.4f}, Val feature loss: {val_feature_loss:.4f}, Val Base Model Accuracy: {val_base_acc:.4f}, Val CLIP Accuracy: {val_clip_acc:.4f}")


        # Update and save training plots
        plt.figure(figsize=(12, 10))  # Adjust the size if needed

        plt.subplot(2, 2, 1)  # 2 rows, 2 columns, position 1
        plt.grid(True)
        plt.plot(train_losses, '-o', label='Training')
        plt.plot(val_losses, '-o', label='Validation')
        plt.title('Total Loss')
        plt.legend()

        plt.subplot(2, 2, 2)  # 2 rows, 2 columns, position 2
        plt.grid(True)
        plt.plot(train_accs, '-o', label='Training')
        plt.plot(val_accs, '-o', label='Validation')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)  # 2 rows, 2 columns, position 2
        plt.grid(True)
        plt.plot(train_image_losses, '-o', label='Train CLIP Image Loss')
        plt.plot(val_image_losses, '-o', label='Val CLIP Image Loss')
        plt.title('CLIP Image Loss')
        plt.legend()

        plt.subplot(2, 2, 4)  # 2 rows, 2 columns, position 3
        plt.grid(True)
        plt.plot(train_text_losses, '-o', label='Train CLIP Text Loss')
        plt.plot(val_text_losses, '-o', label='Val CLIP Text Loss')
        plt.title('CLIP Text Loss')
        plt.legend()

        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, 'training_validation_plots.png'))
        plt.close()

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(projector.state_dict(), os.path.join(save_dir, "best_projector_weights.pth"))
            if not args.use_default_prompt:
                torch.save(mapping_sequence, os.path.join(save_dir, "best_mapping_sequence.pth"))

    # Save the trained Visual Transformer model
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector_weights.pth"))
    if not args.use_default_prompt:
        torch.save(mapping_sequence, os.path.join(save_dir, "projector_mapping_sequence.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--data_dir', type=str, default='data/domainnet_v1.0', help='Path to the data directory')
    parser.add_argument('--domain', type=str, required=True, help='Name of the domain to load')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to (assumes square images)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--class_percentage', type=float, default=1, help='Percentage of classes to be included (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--resnet_model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='resnet18', help='Type of ResNet model to use')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint to resume training from')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer')
    parser.add_argument('--resnet_dim', type=int, default=2048, help='Dimension of the ResNet embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--teacher_temp', type=float, default=0.5, help='Temperature for Dino loss')
    parser.add_argument('--student_temp', type=float, default=1, help='Temperature for Dino loss')
    parser.add_argument('--prompt_embeddings_pth', type=str, required=True, help='Path to the prompt embeddings')
    parser.add_argument('--use_default_prompt', type=bool, default=False, help='Use the default prompt instead of FLM')
    parser.add_argument('--mapping_num', type=int, default=1, help='Number of labels to map to each prompt')
    parser.add_argument('--mapping_interval', type=int, default=1, help='Number of epochs between label mapping')
    parser.add_argument('--similarity_mode', type=str, choices=['cosine', 'DN', 'DN*'], default='cosine', help='Type of similarity to use for label mapping')
    parser.add_argument('--feature_similarity', type=bool, default=True, help='Use feature similarity loss')
    parser.add_argument('--feature_sim_weight', type=float, default=0, help='Weight for feature similarity loss')
    parser.add_argument('--distill_loss_weight', type=float, default=1, help='Weight for distillation loss')

    args = parser.parse_args()

    # Print the arguments
    print(args)

    main(args)

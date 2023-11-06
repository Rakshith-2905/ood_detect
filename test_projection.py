import torch
import torch.nn.functional as F
import torch.nn as nn
import clip

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial


from models.resnet import CustomResNet
from models.visual_transformer import ProjectionHead, VisualTransformer
from domainnet_data import DomainNetDataset, get_domainnet_loaders, get_data_from_saved_files
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow, plot_confusion_matrix
from prompts.FLM import generate_label_mapping_by_frequency, label_mapping_base

def get_all_domainnet_loaders(batch_size=32):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"] 
    loaders_dict = {}
    for domain in domains:
        loaders_dict[domain], class_names = get_domainnet_loaders(domain, batch_size=batch_size)
    return loaders_dict, class_names

def evaluate(val_loader, resnet_model, projector, text_encodings, criterion, device, label_mapping=None):
    resnet_model.eval()
    projector.eval()
    
    total_loss = 0
    total_base_model_acc = 0
    total_clip_acc = 0
    total_gt_clip_acc = 0

    all_preds_resnet = []
    all_preds_proj = []
    all_labels = []

    total_samples = len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating')
    with torch.no_grad():
        for images, labels in pbar:

            images, labels = images.to(device), labels.to(device)
            
            # Compute the ResNet embeddings
            resnet_logits, resnet_embeddings = resnet_model(images, return_features=True)
            probs_from_resnet = F.softmax(resnet_logits, dim=-1)
            
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)
            
            # Compute similarities between image embeddings and text encodings
            similarities = compute_similarities(proj_embeddings, text_encodings, mode=args.similarity_mode)


            if label_mapping is not None:
                similarities = label_mapping(similarities)

            probs_from_proj = F.softmax(similarities, dim=-1)

            # Convert the probabilities to predicted class indices
            preds_from_resnet = torch.argmax(probs_from_resnet, dim=-1)
            preds_from_proj = torch.argmax(probs_from_proj, dim=-1)

            # Extend the lists for later confusion matrix computation
            all_labels.extend(labels.cpu().numpy())
            all_preds_resnet.extend(preds_from_resnet.cpu().numpy())
            all_preds_proj.extend(preds_from_proj.cpu().numpy())

            loss = criterion(similarities, labels)
            batch_clip_acc = compute_accuracy(probs_from_proj, labels)

            total_loss += loss.item() 
            total_clip_acc += batch_clip_acc
    
    # print(f"GT CLIP Acc: {total_gt_clip_acc/len(val_loader)}")
    # assert False
    return total_loss/len(val_loader), total_clip_acc/len(val_loader), all_preds_resnet, all_preds_proj, all_labels

def main(args):

    base_dir = f"logs/classifier/imagenet/{args.resnet_model}_{args.dataset}_{args.domain}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class names from a text file
    with open('data/domainnet_v1.0/class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Get all the test loaders for each domain
    loaders_dict, class_names = get_all_domainnet_loaders(batch_size=args.batch_size)

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

    if not args.use_default_prompt:
        pass
        # mapping_sequence = torch.load()
        # label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    else:
        label_mapping = None

    prompt_embeddings = torch.load(args.prompt_embeddings_pth)

    if args.use_default_prompt:
        text_encodings = prompt_embeddings[0]
    else:
        # Merge all the prompt embeddings into one tensor
        text_encodings = torch.cat(prompt_embeddings, dim=0)

    print(f"Loaded text encodings of shape: {text_encodings.shape}")

    # save directory is the director of the projector checkpoint
    save_dir = os.path.dirname(args.projector_checkpoint_path)
    save_dir = os.path.join(save_dir, 'evaluation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    for domain, loader in loaders_dict.items():

        if domain != 'real':
            continue
        val_loader = loader['test']
        loss, acc, resnet_pred, proj_pred, gt_labels = evaluate(val_loader, resnet_model, projector, text_encodings, criterion, device, label_mapping=label_mapping)
        results[domain] = {"Loss": loss, "Accuracy": acc}
        print(f"Domain: {domain}\tLoss: {loss:.4f}\tAccuracy: {acc:.4f}%")

        # Compute the confusion matrix
        if domain == 'real':
            plot_confusion_matrix(proj_pred, resnet_pred, class_names, save_dir)

    assert False
    avg_ood_acc = 0
    for domain, acc in results.items():
        if domain != 'real':
            avg_ood_acc += acc['Accuracy']
    avg_ood_acc /= 5
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Domain\tLoss\tAccuracy\n")
        for domain, metrics in results.items():
            f.write(f"{domain}\t{metrics['Loss']:.4f}\t{metrics['Accuracy']:.4f}%\n")
        f.write(f"Average OOD\t{avg_ood_acc:.4f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--dataset', type=str, required=True, help='Name of the WILDS dataset')
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

    # Print the arguments
    print(args)

    main(args)

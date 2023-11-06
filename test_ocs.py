import torch
import torch.nn.functional as F
import torch.nn as nn
import clip
import skimage as sk
from skimage.filters import gaussian
import numpy as np
from PIL import Image

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from torchvision import transforms


from models.resnet import CustomResNet
from models.visual_transformer import ProjectionHead, VisualTransformer
from domainnet_data import DomainNetDataset, get_domainnet_loaders, get_data_from_saved_files
from utils import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow, plot_confusion_matrix
from prompts.FLM import generate_label_mapping_by_frequency, label_mapping_base

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def load_image(file_path):
    """
    Load an image and convert it to a NumPy array with values in the range [0, 255].

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Image as a NumPy array with values in the range [0, 255].
    """
    # Open the image file
    image = Image.open(file_path)

    # Convert to RGB mode if not already in RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to NumPy array
    image_array = np.array(image)

    # Ensure values are in the range [0, 255]
    image_array = np.clip(image_array, 0, 255)

    return image_array

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def impulse_noise(x, severity=1):
    #c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    c = [.03, .09, 0.17, 0.3, 5.0][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    #c = [1, 3, 9, 11, 15][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def get_all_domainnet_loaders(batch_size=32):
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"] 
    loaders_dict = {}
    for domain in domains:
        loaders_dict[domain], class_names = get_domainnet_loaders(domain, batch_size=batch_size)
    return loaders_dict, class_names

def evaluate(val_loader, resnet_model, projector, text_encodings, criterion, clip_model, device, label_mapping=None):
    resnet_model.eval()
    projector.eval()
    
    total_loss = 0
    total_base_model_acc = 0
    total_clip_acc = 0
    total_gt_clip_acc = 0

    all_preds_resnet = []
    all_preds_proj = []
    all_labels = []

    all_probs_resnet = []
    all_probs_proj = []

    total_samples = len(val_loader.dataset)
    
    # Wrap the val_loader with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f'Validating')
    with torch.no_grad():
        for images, labels in pbar:

            images, labels = images.to(device), labels.to(device)
            
            # Compute the ResNet embeddings
            resnet_logits, resnet_embeddings = resnet_model(images, return_features=True)
            probs_from_resnet = F.softmax(resnet_logits, dim=-1)
            all_probs_resnet.append(probs_from_resnet)
            
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)
            
            # Compute similarities between image embeddings and text encodings
            similarities = compute_similarities(proj_embeddings, text_encodings, mode=args.similarity_mode)*100.
            #norm_emb = proj_embeddings/proj_embeddings.norm(dim=-1, keepdim=True)
            #similarities = clip_model.logit_scale.exp().float() * norm_emb.float() @ text_encodings.t().float()


            if label_mapping is not None:
                similarities = label_mapping(similarities)

            probs_from_proj = F.softmax(similarities, dim=-1)
            all_probs_proj.append(probs_from_proj)

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
    return total_loss/len(val_loader), total_clip_acc/len(val_loader), all_preds_resnet, all_preds_proj, all_labels, torch.cat(all_probs_resnet,0), torch.cat(all_probs_proj,0)

def main(args):

    base_dir = f"logs/classifier/{args.resnet_model}_{args.dataset}_{args.domain}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load class names from a text file
    with open('./data/domainnet_v1.0/class_names.txt', 'r') as f:
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
    save_dir = os.path.join(save_dir, 'evaluation_ocs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    results = {}


    # clip_model, preprocess = clip.load("RN50")
    # convert_models_to_fp32(clip_model)
    # sample_image_paths = ['./data/domainnet_v1.0/real/ant/real_007_000204.jpg']#, '../data/domainnet/real/airplane/real_002_000201.jpg']
    # images = []
    # corr = 'gaussian_blur'
    # severities = [0,1,2,3,4]
    # l = [6]*(len(severities)+2)
    # transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    # for s in sample_image_paths:
    #     im = load_image(s)
    #     images.append(transform(Image.fromarray(im.astype(np.uint8))))
    #     images[-1] = images[-1].unsqueeze(0)
    #     for severity in severities:
    #         if corr == 'impulse_noise':
    #             images.append(transform(Image.fromarray(impulse_noise(im, severity=severity).astype(np.uint8))))
    #             images[-1] = images[-1].unsqueeze(0)
    #         elif corr == 'gaussian_blur':
    #             images.append(transform(Image.fromarray(gaussian_blur(im, severity=severity).astype(np.uint8))))
    #             images[-1] = images[-1].unsqueeze(0)
    #         elif corr == 'gaussian_noise':
    #             images.append(transform(Image.fromarray(gaussian_noise(im, severity=severity).astype(np.uint8))))
    #             images[-1] = images[-1].unsqueeze(0)
    
    # images.append(1.0*torch.randn_like(images[-1]))

    # images = torch.cat(images, 0)
    # print(images.shape)
    # l = torch.from_numpy(np.array(l))
    # valset = torch.utils.data.TensorDataset(images, l)
    # val_loader = torch.utils.data.DataLoader(valset, batch_size=7, shuffle=False)

    # loss, acc, resnet_pred, proj_pred, gt_labels, resnet_probs, proj_probs = evaluate(val_loader, resnet_model, projector, text_encodings, criterion, clip_model, device, label_mapping=label_mapping)
    # resnet_probs = resnet_probs.cpu().data.numpy()
    # proj_probs = proj_probs.cpu().data.numpy()

    # print(f'Resnet Preds = {np.argmax(resnet_probs,1)}')
    # print(f'Proj. Preds = {np.argmax(proj_probs,1)}')
    # print(resnet_probs.shape)

    # fig, axs = plt.subplots(2, 7, figsize=(18, 6))
    # axs = axs.flatten()
    # # Plot stem plots in each subplot
    # for i in range(len(axs)):
    #     if i <7:
    #         ax = axs[i]
    #         ax.stem(range(len(resnet_probs[i,:])), resnet_probs[i,:], linefmt='b-', markerfmt='bo', basefmt=' ', use_line_collection=True)
    #         if i==0:
    #             ax.set_title('Clean')
    #         elif i>0 and i<6:
    #             ax.set_title(f'{corr} severity {i}')
    #         else:
    #             ax.set_title('Noise')
    #         ax.set_ylim(0,1.1)
    #     else:
    #         ax = axs[i]
    #         ax.stem(range(len(proj_probs[i-7,:])), proj_probs[i-7,:], linefmt='b-', markerfmt='bo', basefmt=' ', use_line_collection=True)
    #         if i ==7:
    #             ax.set_title('Clean')
    #         elif i>7 and i<13:
    #             ax.set_title(f'{corr} severity {i-7}')
    #         else:
    #             ax.set_title('Noise')
    #         ax.set_ylim(0,1.1)


    # # Adjust layout and show the plot
    # plt.tight_layout()

    # plt.savefig(f'{save_dir}/ocs-test-{corr}_{sample_image_paths[0].split("/")[-3]}_{sample_image_paths[0].split("/")[-2]}_{sample_image_paths[0].split("/")[-1]}.png', bbox_inches='tight')



    with open('/usr/workspace/viv41siv/CVPR2024/failure-detection/ood_detect/data/domainnet_v1.0/text_files/real_test.txt') as f:
        lines = [line.strip() for line in f.readlines()]
    f.close()
    idx = list(np.random.choice(range(len(lines)), 1, replace=False))
    sample_image_paths = [f'./data/domainnet_v1.0/{lines[i].split(" ")[0]}' for i in range(len(lines))]
    sample_image_paths = [sample_image_paths[i] for i in idx]
    sample_image_labels = [int(lines[i].split(" ")[1]) for i in range(len(lines))]
    sample_image_labels = [sample_image_labels[i] for i in idx]*10
    clip_model, preprocess = clip.load("RN50")
    convert_models_to_fp32(clip_model)
    
    corr = 'impulse_noise'
    severities = [-1,0,1,2,3,4]
    transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    resnet_probs_severities, proj_probs_severities = [], []
    for severity in severities:
        print(f'Severity {severity} - {corr}')
        images = []
        for i, s in enumerate(sample_image_paths):
            print(f'Processing Image {i}')
            im = load_image(s)
            for j in range(10):
                if severity == -1:
                    images.append(transform(Image.fromarray(im.astype(np.uint8))))
                    images[-1] = images[-1].unsqueeze(0)
                else:
                    if corr == 'impulse_noise':
                        images.append(transform(Image.fromarray(impulse_noise(im, severity=severity).astype(np.uint8))))
                        images[-1] = images[-1].unsqueeze(0)
                    elif corr == 'gaussian_blur':
                        images.append(transform(Image.fromarray(gaussian_blur(im, severity=severity).astype(np.uint8))))
                        images[-1] = images[-1].unsqueeze(0)
                    elif corr == 'gaussian_noise':
                        images.append(transform(Image.fromarray(gaussian_noise(im, severity=severity).astype(np.uint8))))
                        images[-1] = images[-1].unsqueeze(0)
    
        images = torch.cat(images, 0)
        print(images.shape)
        l = torch.from_numpy(np.array(sample_image_labels))
        valset = torch.utils.data.TensorDataset(images, l)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)

        loss, acc, resnet_pred, proj_pred, gt_labels, resnet_probs, proj_probs = evaluate(val_loader, resnet_model, projector, text_encodings, criterion, clip_model, device, label_mapping=label_mapping)
        resnet_probs = resnet_probs.cpu().data.numpy()
        proj_probs = proj_probs.cpu().data.numpy()

        #print(f'Resnet Preds = {np.argmax(resnet_probs,1)}')
        #print(f'Proj. Preds = {np.argmax(proj_probs,1)}')
        print(resnet_probs.shape)

        resnet_probs_severities.append(np.mean(resnet_probs,0))
        proj_probs_severities.append(np.mean(proj_probs,0))
    resnet_probs_severities = np.array(resnet_probs_severities)
    proj_probs_severities = np.array(proj_probs_severities)
    print(resnet_probs_severities.shape, proj_probs_severities.shape)
    print(f'Resnet Preds = {np.argmax(resnet_probs_severities,1)}')
    print(f'Proj. Preds = {np.argmax(proj_probs_severities,1)}')
    print(f'True Label = {sample_image_labels[0]}')
    

    fig, axs = plt.subplots(2, 6, figsize=(18, 6))

    # Flatten the 2D subplot array for easy iteration
    axs = axs.flatten()

    # Plot stem plots in each subplot
    for i in range(len(axs)):
        if i <6:
            ax = axs[i]
            ax.stem(range(len(resnet_probs_severities[i,:])), resnet_probs_severities[i,:], linefmt='b-', markerfmt='bo', basefmt=' ', use_line_collection=True)
            if i==0:
                ax.set_title('Clean')
            elif i>0 and i<6:
                ax.set_title(f'{corr} severity {i}')
            #else:
            #    ax.set_title('Noise')
            ax.set_ylim(0,1.1)
        else:
            ax = axs[i]
            ax.stem(range(len(proj_probs_severities[i-6,:])), proj_probs_severities[i-6,:], linefmt='b-', markerfmt='bo', basefmt=' ', use_line_collection=True)
            if i ==6:
                ax.set_title('Clean')
            elif i>6 and i<13:
                ax.set_title(f'{corr} severity {i-6}')
            #else:
            #    ax.set_title('Noise')
            ax.set_ylim(0,1.1)


    # Adjust layout and show the plot
    plt.tight_layout()

    plt.savefig(f'{save_dir}/ocs-test-{corr}_domain_real.png', bbox_inches='tight')







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

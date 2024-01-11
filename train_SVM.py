import os
import sys
import copy
import time
try:
    del os.environ['OMP_PLACES']
    del os.environ['OMP_PROC_BIND']
except:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm


import clip
import csv
from tqdm import tqdm
import numpy as np
import random
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

from train_task_distillation import build_classifier, get_CLIP_text_encodings, get_dataset_from_file, get_dataset
from models.projector import ProjectionHead
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_save_dir(args):
    
    save_dir = os.path.dirname(args.step1_checkpoint_path)
    save_dir = os.path.join(save_dir, 'failure_detector')

    return save_dir

@torch.no_grad()
def evaulate(data_loader, clip_model, classifier,
                img_projector, text_projector, text_encodings_raw, class_prompts,
                clip_prompted_txt_enc, clip_prompted_img_enc, criterion, epoch):

    clip_model.eval()
    classifier.eval()

    def set_eval_mode(*models):
        for model in models:
            if model:
                model.eval()

    set_eval_mode(img_projector, text_projector, clip_prompted_txt_enc, clip_prompted_img_enc)

    classifier_not_correct = []
    proj_not_correct = []

    classifier_features = []
    clip_features = []
    proj_features = []

    total_base_model_acc = 0
    total_plumber_acc = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch} Val')
    for images_batch, labels, images_clip_batch in pbar:

        images_batch = images_batch.to(args.device)
        images_clip_batch = images_clip_batch.to(args.device)
        labels = labels.to(args.device)
        
        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        classifier_features.extend(classifier_embeddings.cpu().numpy())

        # Learnable Image Prompting
        if clip_prompted_img_enc:
            clip_image_embeddings = clip_prompted_img_enc(images_clip_batch) # (batch_size, embedding_dim)
            clip_features.extend(clip_model.encode_image(images_clip_batch).cpu().numpy())
        else:
            clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)
            clip_features.extend(clip_model.encode_image(images_clip_batch).cpu().numpy())

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        # Project the image embeddings
        if img_projector:
            if args.proj_clip: # this is PLUMBER
                proj_embeddings = img_projector(clip_image_embeddings) # (batch_size, projection_dim)
            else: # this is LIMBER
                proj_embeddings = img_projector(classifier_embeddings) # (batch_size, projection_dim)
        else:
            proj_embeddings = clip_image_embeddings

        # Learnable prompts for the text prompts
        if clip_prompted_txt_enc:
            text_encodings_raw = clip_prompted_txt_enc(class_prompts)
            text_encodings_raw = text_encodings_raw.type(torch.float32)

        # Project the text embeddings
        if text_projector:
            text_encodings = text_projector(text_encodings_raw)
        else:
            text_encodings = text_encodings_raw
            
        normalized_proj_embeddings = F.normalize(proj_embeddings, dim=-1)
        normalized_text_encodings = F.normalize(text_encodings, dim=-1)# (num_classes, projection_dim)

        # make the text embeddings to the same data type as image embeddings
        normalized_text_encodings = normalized_text_encodings.type_as(normalized_proj_embeddings)
        # T100 is the logits scale from CLIP
        logits_projection = 100*normalized_proj_embeddings @ normalized_text_encodings.t() # (batch_size, num_classes)

        probs_from_classifier = F.softmax(classifier_logits, dim=-1)
        probs_from_proj = F.softmax(logits_projection, dim=-1)

        batch_base_model_acc = compute_accuracy(probs_from_classifier, labels)
        batch_plumber_acc = compute_accuracy(probs_from_proj, labels)

        total_base_model_acc += batch_base_model_acc
        total_plumber_acc += batch_plumber_acc  

        # Save the features and the failure labels
        # 1 if prediction is incorrect(Failure), 0 otherwise
        classifier_not_correct.extend((probs_from_classifier.argmax(dim=-1) != labels).cpu().numpy())
        proj_not_correct.extend((probs_from_proj.argmax(dim=-1) != labels).cpu().numpy())
        proj_features.extend(proj_embeddings.cpu().numpy())


    # Compute the average accuracy
    total_base_model_acc /= len(data_loader)
    total_plumber_acc /= len(data_loader)

    features = {
        'classifier_features': classifier_features,
        'clip_features': clip_features,
        'proj_features': proj_features
    }
    failure_labels = {
        'classifier_not_correct': classifier_not_correct,
        'proj_not_correct': proj_not_correct
    }

    return total_base_model_acc, total_plumber_acc, features, failure_labels

def subsample_features_labels(features, labels):
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Find the indices of the failure and success cases
    failure_indices = np.where(labels == 1)[0]
    success_indices = np.where(labels == 0)[0]

    # Count the number of failures and successes
    num_failures = len(failure_indices)
    num_successes = len(success_indices)

    # If there are more successes than failures, undersample the successes
    # Otherwise, undersample the failures
    if num_successes > num_failures:
        undersample_indices = np.random.choice(success_indices, size=num_failures, replace=False)
        failure_indices = np.random.choice(failure_indices, size=num_failures, replace=False)
        indices = np.concatenate([undersample_indices, failure_indices])
        print(f"Local Subpop: Subsampling performed. Reduced the number of successes from {num_successes} to {num_failures}.")
    else:
        undersample_indices = np.random.choice(failure_indices, size=num_successes, replace=False)
        success_indices = np.random.choice(success_indices, size=num_successes, replace=False)
        indices = np.concatenate([undersample_indices, success_indices])
        print(f"Local Subpop: Subsampling performed. Reduced the number of failures from {num_failures} to {num_successes}.")

    # Use the chosen indices to select the features and labels
    subsampled_features = features[indices]
    subsampled_labels = labels[indices]

    print(f"Local Subpop: The total number of samples has been reduced from {len(labels)} to {len(subsampled_labels)}.")

    return list(subsampled_features), list(subsampled_labels)

def compute_and_plot_metrics(true_labels, preds, preds_proba, log_dir):
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(filename=os.path.join(log_dir, 'metrics.log'), level=logging.INFO)
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, preds)

    # Plot confusion matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('failure_Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'failure_confusion_matrix.png'))
    plt.show()

    # Compute metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    auc_roc = roc_auc_score(true_labels, preds_proba)
    mcc = matthews_corrcoef(true_labels, preds) # Matthews correlation coefficient

    # Print and log metrics
    metrics = f'Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%, AUC-ROC: {auc_roc*100:.2f}%, MCC: {mcc*100:.2f}%'
    print(metrics)
    logging.info(metrics)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, preds_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(log_dir, 'roc_curve.png'))
    plt.show()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, preds_proba)
    average_precision = average_precision_score(true_labels, preds_proba)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(log_dir, 'precision_recall_curve.png'))
    plt.show()

def get_failure_data(args):

    # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.to(args.device)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    classifier = classifier.to(args.device)

    print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    # Load the checkpoint for the step 1 model
    checkpoint = torch.load(args.step1_checkpoint_path, map_location=args.device)

    img_projector = None
    if args.img_projection:
        if args.proj_clip:
            # This is PLUMBER
            img_projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
            print(f"Constructed img emb projection PLUMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
        else:
            # This is LIMBER
            img_projector = ProjectionHead(input_dim=classifier.feature_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
            print(f"Constructed img emb projection LIMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
        img_projector = img_projector.to(args.device)
        img_projector.load_state_dict(checkpoint['img_projector'])

    text_projector = None
    if args.txt_projection:
        text_projector = ProjectionHead(input_dim=args.projection_dim, output_dim=args.projection_dim,is_mlp=args.is_mlp)
        print(f"Constructed text emb projection PLUMBER with projection dim: {args.projection_dim} and is_mlp: {args.is_mlp}")
        text_projector = text_projector.to(args.device)
        text_projector.load_state_dict(checkpoint['text_projector'])
    ########################### Load the dataset ############################

    if args.use_saved_features:
        # data_dir = os.path.join(args.data_dir, args.domain_name)
        # Get the directory of classifier checkpoints
        data_dir = os.path.dirname(args.classifier_checkpoint_path)
        data_dir = os.path.join(data_dir, 'features', args.domain_name)
        train_dataset, val_dataset, class_names = get_dataset_from_file(args.dataset_name, data_dir=data_dir)
        print(f"Using saved features from {args.dataset_name} dataset from {data_dir}")
    else:
        # Create the data loader and wrap them with Fabric
        train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                                data_dir=args.data_dir, clip_transform=clip_transform, img_size=args.img_size, return_failure_set=True)
        print(f"Using {args.dataset_name} dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    failure_loader = torch.utils.data.DataLoader(failure_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)



    print(f"Number of validation examples: {len(val_loader.dataset)}")
    print(f"Number of failure examples: {len(failure_loader.dataset)}")
    print(f"Number of test examples: {len(test_loader.dataset)}")
    # Get the text encodings for the class names
    try:
        text_encodings= torch.load(args.prompt_path)
        if text_encodings.shape[0] != len(class_names):
            raise Exception("Text encodings shape does not match the number of classes")

        print(f"Loaded CLIP {args.clip_model_name} text encodings from {args.prompt_path}, {text_encodings.shape}")
    except:
        assert False, "{args.prompt_path} Prompt file not found."
        # text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        # print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")

    class_prompts = None
    clip_prompted_txt_enc = None
    if args.cls_txt_prompts:

        # Create the prompted CLIP model
        clip_prompted_txt_enc = PromptedCLIPTextEncoder(clip_model, n_ctx=args.n_promt_ctx, num_classes=len(class_names), 
                                                    device=args.device, is_dist_prompt=False)
        clip_prompted_txt_enc = clip_prompted_txt_enc.to(args.device)
        clip_prompted_txt_enc.load_state_dict(checkpoint['clip_prompted_txt_enc'])

        class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
        print(f"Constructed CLIP Class specific Prompted Text Encoder with {len(class_names)} classes")

    elif args.dataset_txt_prompt:
        # Create the prompted CLIP model
        clip_prompted_txt_enc = PromptedCLIPTextEncoder(clip_model, n_ctx=args.n_promt_ctx, num_classes=len(class_names),
                                                    device=args.device, is_dist_prompt=True)
        clip_prompted_txt_enc = clip_prompted_txt_enc.to(args.device)
        clip_prompted_txt_enc.load_state_dict(checkpoint['clip_prompted_txt_enc'])
        
        class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]
        print(f"Constructed CLIP Dataset specific Prompted Text Encoder with {len(class_names)} classes")
    
    clip_prompted_img_enc = None
    if args.img_prompting:
        # Create the prompted CLIP model
        clip_prompted_img_enc = PromptedCLIPImageEncoder(clip_model, num_tokens=args.n_promt_ctx, device=args.device)
        clip_prompted_img_enc = clip_prompted_img_enc.to(args.device)
        clip_prompted_img_enc.load_state_dict(checkpoint['clip_prompted_img_enc'])

        print(f"Constructed CLIP Prompted Image Encoder")

    # Loss function
    criterion = SimpleDINOLoss(student_temp=args.student_temp, teacher_temp=args.teacher_temp)

    val_base_acc, val_plumber_acc, val_features, val_failure_labels = evaulate(
                                                                failure_loader, clip_model, classifier,
                                                                img_projector, text_projector,
                                                                text_encodings, class_prompts,
                                                                clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                criterion, 0)
    
    test_base_acc, test_plumber_acc, test_features, test_failure_labels = evaulate(
                                                                test_loader, clip_model, classifier,
                                                                img_projector, text_projector,
                                                                text_encodings, class_prompts,
                                                                clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                criterion, 0)
    
                        
    output_message = f"Val Base Acc: {val_base_acc:.2f}, Val Plumber Acc: {val_plumber_acc:.2f}, Test Base Acc: {test_base_acc:.2f}, Test Plumber Acc: {test_plumber_acc:.2f}"
    print(output_message)

    return {'failure_dataset': [val_features, val_failure_labels], 'test_dataset': [test_features, test_failure_labels]}

def main(args):
    
    # Load the saved features and labels or create them
    if os.path.exists(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl')):
        with open(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl'), 'rb') as f:
            features_labels_dict = pickle.load(f)
    else:
        features_labels_dict = get_failure_data(args)
        # Save the features and the failure labels
        os.makedirs(os.path.join(args.save_dir, 'features'), exist_ok=True)
        with open(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl'), 'wb') as f:
            pickle.dump(features_labels_dict, f)

    failure_features = features_labels_dict['failure_dataset'][0]['proj_features']
    failure_labels = features_labels_dict['failure_dataset'][1]['classifier_not_correct'] 

    # Subsample the features and labels
    failure_features, failure_labels = subsample_features_labels(failure_features, failure_labels)
    
    test_features = features_labels_dict['test_dataset'][0]['proj_features']
    test_failure_labels = features_labels_dict['test_dataset'][1]['classifier_not_correct']
    
    print(f"Number of failure cases: {sum(failure_labels)}")
    print(f"Number of success cases: {len(failure_labels) - sum(failure_labels)}")
    print(f"Number of failure cases in test set: {sum(test_failure_labels)}")
    print(f"Number of success cases in test set: {len(test_failure_labels) - sum(test_failure_labels)}")


    # Start the timer
    start_time = time.time()
    # Train the SVM
    svm_model = svm.SVC(kernel='rbf', probability=True)
    svm_model.fit(failure_features, failure_labels)

    # Get the prediction probabilities on the test set
    svm_preds_proba = svm_model.predict_proba(test_features)[:, 1]

    # Evaluate the SVM
    svm_preds = svm_model.predict(test_features)
    svm_accuracy = accuracy_score(svm_preds, test_failure_labels)


    # Calculate and print the duration
    duration = time.time() - start_time
    print(f"SVM fitting completed in {duration:.2f} seconds")

    print(f'Failure Detector PLUMBER Test Accuracy: {svm_accuracy * 100:.2f}%')


    # compute_and_plot_metrics(test_failure_labels, svm_preds, svm_preds_proba, log_dir=args.save_dir)
    
    # # Save the SVM model
    # with open(os.path.join(args.save_dir, 'svm_model.pkl'), 'wb') as f:
    #     pickle.dump(svm_model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--severity', type=int, default=3, help='Severity of the corruption')
    parser.add_argument('--num_classes', type=int, default=345, help='Number of classes in the dataset')
    parser.add_argument('--train_on_testset', action='store_true', help='Whether to train on the test set or not')
    parser.add_argument('--use_saved_features',action = 'store_true', help='Whether to use saved features or not')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the dataloader')
    parser.add_argument('--img_size', type=int, default=75, help='Image size for the celebA dataloader only')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    parser.add_argument('--img_projection', action='store_true', help='Whether to use task projection or not')
    parser.add_argument('--txt_projection', action='store_true', help='Whether to use text projection or not')
    parser.add_argument('--img_prompting', action='store_true', help='Whether to use image prompting or not')
    parser.add_argument('--step1_checkpoint_path', type=str, help='Path to checkpoint to load the step 1 model from')

    parser.add_argument('--classifier_name', required=True,  help='Name of the classifier to use sam_vit_h, mae_vit_large_patch16, dino_vits16, resnet50, resnet50_adv_l2_0.1, resnet50_adv_l2_0.5, resnet50x1_bitm, resnetv2_101x1_bit.goog_in21k, deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50, fcn_resnet101')
    parser.add_argument('--classifier_checkpoint_path', type=str, help='Path to checkpoint to load the classifier from')
    parser.add_argument('--use_imagenet_pretrained', action='store_true', help='Whether to use imagenet pretrained weights or not')
    parser.add_argument('--clip_model_name', default='ViT-B/32', help='Name of the CLIP model to use.')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt file')
    parser.add_argument('--n_promt_ctx', type=int, default=16, help='Number of learnable prompt token for each cls')
    parser.add_argument('--cls_txt_prompts', action='store_true', help='Whether to use learnable prompts or not')
    parser.add_argument('--dataset_txt_prompt', action='store_true', help='Whether to use dataset level prompts or class level prompts')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, choices=['adam','adamw', 'sgd'], default='adamw', help='Type of optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the results')
    parser.add_argument('--prefix', type=str, default='', help='prefix to add to the save directory')

    parser.add_argument('--proj_clip', action='store_true', help='Whether to project the clip embeddings or the classifier embeddings')
    parser.add_argument('--projection_dim', type=int, default=512, help='Dimension of the projected embeddings')
    parser.add_argument('--is_mlp', action='store_true', help='Whether to use MLP projection head or not')
    parser.add_argument('--teacher_temp', type=float, default=0.5, help='Temperature for Dino loss')
    parser.add_argument('--student_temp', type=float, default=1, help='Temperature for Dino loss')
    parser.add_argument('--resume_checkpoint_path', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--weight_img_loss', type=float, default=0.5, help='Weight for image loss')
    parser.add_argument('--weight_txt_loss', type=float, default=0.5, help='Weight for text loss')
    
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of gpus for DDP per node')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for DDP')
    parser.add_argument('--template_num', type=int, default=0, help='CLIP text prompt number')

    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    print(args)
    sys.stdout.flush()

    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    os.makedirs(args.save_dir, exist_ok=True)

    seed_everything(args.seed)

    main(args)

'''
python train_SVM.py \
    --data_dir "./data/" \
    --domain_name 'real' \
    --dataset_name "CelebA" \
    --train_on_testset \
    --num_classes 2 \
    --batch_size 256 \
    --seed 42 \
    --img_size 75 \
    --img_projection --txt_projection \
    --classifier_name "resnet18" \
    --classifier_checkpoint_path "logs/CelebA/resnet18/classifier/checkpoint_29.pth" \
    --step1_checkpoint_path "logs/CelebA/resnet18/plumber_img_text_proj/_clsEpoch_29_bs_256_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/step_1/best_projector_weights.pth" \
    --clip_model_name 'ViT-B/32' \
    --prompt_path "data/CelebA/CelebA_CLIP_ViT-B_32_text_embeddings.pth" \
    --n_promt_ctx 16 \
    --num_epochs 10 \
    --optimizer 'sgd' \
    --learning_rate 0.1 \
    --val_freq 1 \
    --prefix '' \
    --proj_clip \
    --projection_dim 512 \
    --teacher_temp 2 \
    --student_temp 1 \
    --weight_img_loss 1 \
    --weight_txt_loss 1 \
    --num_gpus 1 \
    --num_nodes 1
'''
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
import json

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from train_task_distillation import build_classifier, get_CLIP_text_encodings, get_dataset_from_file, get_dataset
from models.projector import ProjectionHead
from models.prompted_CLIP import PromptedCLIPTextEncoder, PromptedCLIPImageEncoder
from utils_proj import SimpleDINOLoss, compute_accuracy, compute_similarities, plot_grad_flow

from models import svm_wrapper
from models.plumber import PLUMBER
from data_utils.FD_caption_set import get_caption_set

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_save_dir(args):
    
    save_dir = os.path.dirname(args.step1_checkpoint_path)
    save_dir = os.path.join(save_dir, 'failure_detector')

    if args.dataset_name in ['cifar10-c', 'cifar100-c', 'imagenet-c']:
        severity = args.severity if args.severity else ''
        save_dir = os.path.join(save_dir, f'{args.dataset_name}_{args.domain_name}_{severity}')
    elif args.dataset_name == 'domainnet':
        save_dir = os.path.join(save_dir, f'{args.domain_name}')

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

    gt_labels = []
    classifier_preds = []
    proj_preds = []

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

        clip_raw_embeddings = clip_model.encode_image(images_clip_batch).float() # (batch_size, embedding_dim)

        # Learnable Image Prompting
        if clip_prompted_img_enc:
            clip_image_embeddings = clip_prompted_img_enc(images_clip_batch) # (batch_size, embedding_dim)
        else:
            clip_image_embeddings = clip_model.encode_image(images_clip_batch) # (batch_size, embedding_dim)

        clip_image_embeddings = clip_image_embeddings.type_as(classifier_embeddings)

        # Project the image embeddings if no projection use the clip embeddings
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

        # # Save the features and the failure labels
        # # 1 if prediction is incorrect(Failure), 0 otherwise
        # classifier_not_correct.extend((probs_from_classifier.argmax(dim=-1) != labels).cpu().numpy())
        # proj_not_correct.extend((probs_from_proj.argmax(dim=-1) != labels).cpu().numpy())

        classifier_preds.extend(probs_from_classifier.argmax(dim=-1).cpu().numpy())
        proj_preds.extend(probs_from_proj.argmax(dim=-1).cpu().numpy())
        gt_labels.extend(labels.cpu().numpy())

        classifier_features.extend(classifier_embeddings.type(torch.float32).cpu().numpy())
        clip_features.extend(clip_raw_embeddings.cpu().numpy())
        proj_features.extend(proj_embeddings.cpu().numpy())


    # Compute the average accuracy
    total_base_model_acc /= len(data_loader)
    total_plumber_acc /= len(data_loader)

    features = {
        'classifier_features': classifier_features,
        'clip_features': clip_features,
        'proj_features': proj_features
    }
    predictions = {
        'gt_labels': gt_labels,
        'classifier_preds': classifier_preds,
        'proj_preds': proj_preds
    }

    return total_base_model_acc, total_plumber_acc, features, predictions

def get_features_preds(args):

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
                                                                data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                img_size=args.img_size, return_failure_set=True,
                                                                domain_name=args.domain_name, severity=args.severity)
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

    train_base_acc, train_plumber_acc, train_features, train_preds = evaulate(
                                                                train_loader, clip_model, classifier,
                                                                img_projector, text_projector,
                                                                text_encodings, class_prompts,
                                                                clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                criterion, 0)

    val_base_acc, val_plumber_acc, val_features, val_preds = evaulate(
                                                                failure_loader, clip_model, classifier,
                                                                img_projector, text_projector,
                                                                text_encodings, class_prompts,
                                                                clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                criterion, 0)
    
    test_base_acc, test_plumber_acc, test_features, test_preds = evaulate(
                                                                test_loader, clip_model, classifier,
                                                                img_projector, text_projector,
                                                                text_encodings, class_prompts,
                                                                clip_prompted_txt_enc, clip_prompted_img_enc,
                                                                criterion, 0)
    
                        
    output_message = f"Val Base Acc: {val_base_acc:.2f}, Val Plumber Acc: {val_plumber_acc:.2f}, Test Base Acc: {test_base_acc:.2f}, Test Plumber Acc: {test_plumber_acc:.2f}"
    
    return {
        'train': [train_features, train_preds],
        'val': [val_features, val_preds],
        'test': [test_features, test_preds]
    }

def compute_and_plot_metrics(metric_dict, log_dir):
    # Ensure the logging directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Compute confusion matrix
    cm = metric_dict['confusion_matrix']

    # Plot confusion matrix
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('failure_Confusion Matrix')
    plt.savefig(os.path.join(log_dir, 'failure_confusion_matrix.png'))
    plt.show()

    true_labels = metric_dict['ytrue']
    preds = metric_dict['ypred']
    # Compute metrics
    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds) # Matthews correlation coefficient

    # Print and log metrics
    metrics = f'{args.svm_features}\tAccuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1 Score: {f1*100:.2f}%'
    print(metrics, "\n\n")
    with open(os.path.join(log_dir, 'metrics.txt'), 'a') as f:
        f.write(metrics)
        f.write("\n")

def get_caption_scores_(plumber, captions, reference_caption, svm_fitter):
    caption_latent = plumber.encode_text_batch(captions)
    reference_latent = plumber.encode_text_batch([reference_caption])[0]
    latent = caption_latent - reference_latent

    out_mask = svm_fitter.predict(latent)
    decisions = svm_fitter.decision_function(latent)

    return decisions, caption_latent, out_mask

failure_caption_data_name_map = {
    'cifar10': 'CIFAR',
    'cifar10-c': 'CIFAR',
    'cifar10-limited': 'CIFAR',
    'CelebA': 'CELEBA',
    'cifar100': 'CIFAR100',
    'imagenet': 'IMAGENET',
}
def caption_failure_modes(args, svm_fitter, plumber):
    data_name = failure_caption_data_name_map[args.dataset_name]
    captions = get_caption_set(data_name)

    clip_model, clip_transform = clip.load(args.clip_model_name)

    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, None, None, 
                                                                data_dir=args.data_dir, clip_transform=clip_transform, 
                                                                img_size=args.img_size, domain_name=args.domain_name, severity=args.severity,
                                                                return_failure_set=True)
    

    print(f"Using {args.dataset_name} dataset")

    captions_set_all = []
    if args.dataset_name == 'CelebA':
        class_names = ["person", "person"]
        captions_set_all.extend(captions[class_names[0]]['all'])
    else:
        for i in range(len(class_names)):
            captions_set_all.extend(captions[class_names[i]]['all'])

        
    selected_captions = []    
    topk_caption_decisions_cls = {}

    for target_c in range(len(class_names)):
        target_c_name = class_names[target_c]
        caption_set = captions[target_c_name]['all'] # All captions noun adjective prepositions for the target class
        reference = captions['reference'][target_c] # Class prompts for the target class
        # decisions, _, mask = svm_wrapper.get_caption_scores(plumber=plumber,
        #                                               captions=caption_set,
        #                                               reference_caption=reference,
        #                                               svm_fitter=svm_fitter,
        #                                               target_c=target_c)
        
        decisions, _, mask = get_caption_scores_(plumber, caption_set, reference, svm_fitter)
        
        # Sort the captions based on the decisions
        sorted_indices = np.argsort(decisions)
        sorted_decisions = decisions[sorted_indices]
        sorted_captions = np.asarray(caption_set)[sorted_indices]
        
        sorted_captions = sorted_captions.tolist()
        sorted_decisions = sorted_decisions.tolist()

        print(f"Number of unique decisions: {np.unique(mask)}")
        topk_caption_decisions_cls[target_c] = {
            'Failure': {
                'captions': sorted_captions[:10],
                'decisions': sorted_decisions[:10]
            },
            'Success': {
                'captions': sorted_captions[-10:],
                'decisions': sorted_decisions[-10:]
            }
        }

        selected_captions.append((
            caption_set[np.argmin(decisions)],
            caption_set[np.argmax(decisions)], np.min(decisions), np.max(decisions)))
    
    # Save the topk_caption_decisions_cls captions
    with open(os.path.join(args.save_dir, f'{args.svm_features}_topk_caption_decisions_single_svm.json'), 'w') as f:
        json.dump(topk_caption_decisions_cls, f)
    print(topk_caption_decisions_cls)
    print(f"Selected captions: {selected_captions}")

def learn_svm(val_features, val_labels, val_preds, test_features, test_labels, test_preds, args):

    # Number of unique values in val_preds
    print(f"Unique values in val_preds: {np.unique(val_preds)}")

    # Compare val_preds and val_gt and get the indices where they are equal
    val_correct = (val_preds == val_labels).astype(np.int)
    test_correct = (test_preds == test_labels).astype(np.int)

    print("Number of validation samples: ", len(val_correct))
    print("Number of test samples: ", len(test_correct))

    # Scale the features
    scaler = StandardScaler()
    scaler.fit(val_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf.fit(val_features, val_correct)

    test_correct_preds = clf.predict(test_features)

    test_accuracy = accuracy_score(test_correct,test_correct_preds)
    test_recall = recall_score(test_correct, test_correct_preds)
    test_precision = precision_score(test_correct, test_correct_preds)
    test_f1 = f1_score(test_correct, test_correct_preds)
    
    # Failure estimation report (class 1 represents correct prediction of task model)
    class_report = classification_report(test_correct, test_correct_preds, output_dict=True)

    cm = confusion_matrix(test_correct, test_correct_preds)
    class_level_acc = cm.diagonal()/cm.sum(axis=1)

    estimated_accuracy = np.sum(test_correct_preds)/len(test_correct_preds) # Estimated accuracy from SVM for correct prediction
    task_model_accuracy = np.sum(test_correct)/len(test_correct) # Actuall prediction accuracy
    estimation_gap = np.abs(task_model_accuracy-estimated_accuracy)

    print(len(test_correct_preds), len(test_correct))
    out_metrics = {
        'gt_labels': test_labels,
        'task_pred': test_preds,
        'correct_svm_pred': test_correct_preds,
        'accu_success_pred': class_level_acc[1],
        'accu_failure_pred': class_level_acc[0],
        'confusion_matrix': cm,
        'estimation_gap': estimation_gap,
        'class_report': class_report
    }

    # Convert all NumPy arrays to lists
    for key, value in out_metrics.items():
        if isinstance(value, np.ndarray):
            out_metrics[key] = value.tolist()

    # Save the JSON
    with open(os.path.join(args.save_dir, f'{args.svm_features}_metrics_single_svm.json'), 'w') as f:
        json.dump(out_metrics, f)

    # Print and log metrics
    metrics = f'{args.svm_features}\tAccuracy: {test_accuracy*100:.2f}%, Precision: {test_precision*100:.2f}%, Recall: {test_recall*100:.2f}%, F1 Score: {test_f1*100:.2f}%'
    
    print(metrics, "\n\n")
    with open(os.path.join(args.save_dir, 'single_svm_metrics.txt'), 'a') as f:
        f.write(metrics)
        f.write("\n")
    
    return clf

def main(args):
    
    # # Load the saved features and labels or create them
    if os.path.exists(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl')):
        with open(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl'), 'rb') as f:
            features_pred_dict = pickle.load(f)
    else:
        features_pred_dict = get_features_preds(args)
        # Save the features and the failure labels
        os.makedirs(os.path.join(args.save_dir, 'features'), exist_ok=True)
        with open(os.path.join(args.save_dir, 'features', 'features_labels_dict.pkl'), 'wb') as f:
            pickle.dump(features_pred_dict, f)
    
    train_features = features_pred_dict['train'][0][args.svm_features]
    train_labels = features_pred_dict['train'][1]['gt_labels']
    train_preds = features_pred_dict['train'][1]['classifier_preds']

    val_features = features_pred_dict['val'][0][args.svm_features]
    val_labels = np.asarray(features_pred_dict['val'][1]['gt_labels'])
    val_preds = np.asarray(features_pred_dict['val'][1]['classifier_preds'])

    test_features = features_pred_dict['test'][0][args.svm_features]
    test_labels = np.asarray(features_pred_dict['test'][1]['gt_labels'])
    test_preds = np.asarray(features_pred_dict['test'][1]['classifier_preds'])

    svm_fitter = learn_svm(val_features, val_labels, val_preds, test_features, test_labels, test_preds, args)

    # # Class specific SVM
    # svm_fitter = svm_wrapper.SVMFitter()
    # svm_fitter.set_preprocess(val_features)
    # cv_scores = svm_fitter.fit(preds=val_preds, ys=val_labels, latents=val_features)

    # out_mask, out_decision, metric_dict = svm_fitter.predict(preds=test_preds, ys=test_labels, latents=test_features, compute_metrics=True)

    ## Compute and plot metrics
    # compute_and_plot_metrics(metric_dict, args.save_dir)

    # plumber = PLUMBER(args.clip_model_name, args.num_classes, 
    #                   img_projection=args.img_projection, txt_projection=args.txt_projection, 
    #                   img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, 
    #                   dataset_txt_prompt=args.dataset_txt_prompt, is_mlp=args.is_mlp, device=args.device)
    
    # if args.step1_checkpoint_path:
    #     plumber.load_checkpoint(args.step1_checkpoint_path)

    # caption_failure_modes(args, svm_fitter, plumber)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet on WILDS Dataset')

    parser.add_argument('--data_dir', type=str, default='/usr/workspace/KDML/DomainNet', help='Path to the data directory')
    parser.add_argument('--domain_name', type=str, default='clipart', help='Domain to use for training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='Name of the dataset')
    parser.add_argument('--severity', type=int, help='Severity of the corruption')
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

    parser.add_argument('--svm_features', type=str, default='proj_features', choices=['proj_features', 'clip_features', 'classifier_features'], help='Features to use for SVM')

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
    --num_classes 10 \
    --batch_size 256 \
    --seed 42 \
    --img_size 75 \
    --img_projection \
    --classifier_name "resnet18" \
    --classifier_checkpoint_path "logs/CelebA/resnet18/classifier/checkpoint_29.pth" \
    --step1_checkpoint_path "logs/CelebA/resnet18/plumber_img_proj/_clsEpoch_29_bs_128_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/step_1/best_projector_weights.pth" \
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
    --num_nodes 1 \
    --svm_features 'clip_features'

'''
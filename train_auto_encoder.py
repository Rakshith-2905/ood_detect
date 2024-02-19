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
from delta_uq.deltaUQ import deltaUQ_MLP as uq_wrapper
from models.plumber import PLUMBER
from data_utils.FD_caption_set import get_caption_set

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_save_dir(args):
    
    save_dir = os.path.dirname(args.step1_checkpoint_path)
    save_dir = os.path.join(save_dir, 'AE-failure_detector')

    if args.dataset_name in ['cifar10-c', 'cifar100-c', 'imagenet-c']:
        severity = args.severity if args.severity else ''
        save_dir = os.path.join(save_dir, f'{args.dataset_name}_{args.domain_name}_{severity}')
    elif args.dataset_name == 'domainnet' or args.dataset_name in ['NICOpp']:
        save_dir = os.path.join(save_dir, f'{args.domain_name}')
    
    if args.use_uq_wrapper:
        save_dir = os.path.join(save_dir, 'uq_AE_2_layer')
    else:
        save_dir = os.path.join(save_dir, 'AE')    

    return save_dir

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, min_embed_dim, output_dim, activation='relu', use_batch_norm=False, use_uq_wrapper=False):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Build encoder
        current_dim = input_dim
        if use_uq_wrapper:
            current_dim = input_dim * 2
        layer_idx = 0
        while current_dim > min_embed_dim:
            next_dim = max(min_embed_dim, current_dim // 2)  # Ensure not below min_embed_dim
            self.encoder.add_module(f'encoder_layer_{layer_idx}', 
                                    self._make_layer(activation, current_dim, next_dim, use_batch_norm))
            current_dim = next_dim
            layer_idx += 1

        # When input dim is equal to the min_embd_dim (i.e. no reduction)
        if input_dim == min_embed_dim:
            self.encoder.add_module(f'encoder_layer_{layer_idx}', 
                                    self._make_layer(activation, current_dim, min_embed_dim, use_batch_norm))

        # Build decoder
        layer_idx = 0
        current_dim = min_embed_dim
        while current_dim < output_dim:
            next_dim = min(output_dim, current_dim * 2)  # Ensure not above output_dim
            self.decoder.add_module(f'decoder_layer_{layer_idx}', 
                                    self._make_layer(activation, current_dim, next_dim, use_batch_norm))
            current_dim = next_dim
            layer_idx += 1
        
        # When min embed dim is equal to the output_dim (i.e. no expansion)
        if min_embed_dim == output_dim:
            self.decoder.add_module(f'decoder_layer_{layer_idx}', 
                                    self._make_layer(activation, current_dim, output_dim, use_batch_norm))
            
        # if use_uq_wrapper:
        #     self.encoder = uq_wrapper(self.encoder)

        # merge encoder and decoder
        self.AE = nn.Sequential(
            self.encoder,
            self.decoder
        )

        if use_uq_wrapper:
            self.AE = uq_wrapper(self.AE)

        print("AutoEncoder Architecture: ", self.encoder, self.decoder)

    def _make_layer(self, activation, input_dim, output_dim, use_batch_norm):
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'identity':
            layers.append(nn.Identity())
        else:
            raise NotImplementedError("Only 'relu', 'sigmoid', and 'identity' activations are supported")
        return nn.Sequential(*layers)

    def forward(self, x, anchor=None, n_anchors=1, return_std=False):

        if return_std:
            # encoded, std = self.encoder(x, anchors=anchor, n_anchors=n_anchors, return_std=return_std)
            # decoded = self.decoder(encoded)
            decoded, std = self.AE(x, anchor, n_anchors=n_anchors, return_std=return_std)
            return decoded, std
        
        # encoded = self.encoder(x, anchor, n_anchors=n_anchors)
        # decoded = self.decoder(encoded)

        decoded = self.AE(x, anchor, n_anchors=n_anchors)

        return decoded
    
def train_one_epoch(data_loader, plumber, autoencoder, optimizer,
                    class_prompts, text_encodings_raw, criterion, epoch):

    plumber.set_eval_mode()

    anchor_embeddings = []
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", ncols=0)
    for images_batch, labels, images_clip_batch in pbar:
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if 'clip' in args.svm_features:
            image_encodings = plumber.clip_model.encode_image(images_clip_batch).float()
        else:
            image_encodings = plumber.encode_images(images_clip_batch)


        if len(anchor_embeddings) < 3:
            anchor_embeddings.append(image_encodings)

        ae_image_encodings = autoencoder(image_encodings)

        loss_img = criterion(ae_image_encodings, image_encodings)
        loss_img.backward()
        optimizer.step()

        total_loss += loss_img.item()
        pbar.set_postfix({'Loss': total_loss / (len(pbar) + 1)})

    anchor_embeddings = torch.cat(anchor_embeddings, dim=0)

    return total_loss / len(data_loader), anchor_embeddings

@torch.no_grad()
def evaluate(data_loader, classifier, plumber, autoencoder, anchor_embeddings, clip_model, class_prompts):

    plumber.set_eval_mode()
    autoencoder.eval()
    classifier.eval()

    gt_labels = []
    classifier_preds = []
    proj_preds = []

    classifier_features = []
    clip_features = []
    proj_features = []
    ae_proj_features = []

    total_base_model_acc = 0
    total_plumber_acc = 0
    
    std_list = []
    pbar = tqdm(data_loader, desc=f"Evaluating", ncols=0)
    for images_batch, labels, images_clip_batch in pbar:
        images_batch = images_batch.to(device)
        images_clip_batch = images_clip_batch.to(device)
        labels = labels.to(device)

        classifier_logits, classifier_embeddings = classifier(images_batch, return_features=True) # (batch_size, embedding_dim)

        clip_raw_embeddings = clip_model.encode_image(images_clip_batch).float() # (batch_size, embedding_dim)

        if 'clip' in args.svm_features:
            image_encodings = plumber.clip_model.encode_image(images_clip_batch).float()
        else:
            image_encodings = plumber.encode_images(images_clip_batch)

        ae_image_encodings, stds = autoencoder(image_encodings, anchor=anchor_embeddings, n_anchors=10, return_std=True)
        std_list.append(stds)
        proj_logits = plumber(images_clip_batch, class_prompts)

        total_base_model_acc += compute_accuracy(classifier_logits, labels)
        total_plumber_acc += compute_accuracy(proj_logits, labels)

        gt_labels.extend(labels.cpu().numpy())
        classifier_probs = F.softmax(classifier_logits, dim=1)
        classifier_preds.extend(torch.argmax(classifier_probs, dim=1).cpu().numpy())

        proj_probs = F.softmax(proj_logits, dim=1)
        proj_preds.extend(torch.argmax(proj_probs, dim=1).cpu().numpy())
        
        classifier_features.extend(classifier_embeddings.cpu().numpy())
        clip_features.extend(clip_raw_embeddings.cpu().numpy())
        proj_features.extend(image_encodings.cpu().numpy())
        ae_proj_features.extend(ae_image_encodings.cpu().numpy())

    ae_feature_type = 'proj' if 'proj' in args.svm_features else 'clip'
    features = {
        'classifier_features': classifier_features,
        'clip_features': clip_features,
        'proj_features': proj_features,
        f'ae_{ae_feature_type}_features': ae_proj_features
    }
    predictions = {
        'gt_labels': gt_labels,
        'classifier_preds': classifier_preds,
        'proj_preds': proj_preds
    }

    total_base_model_acc /= len(data_loader)
    total_plumber_acc /= len(data_loader)

    print(f"Base model accuracy: {total_base_model_acc*100:.2f}%, Plumber accuracy: {total_plumber_acc*100:.2f}%")

    # std_summed = torch.cat(std_list, dim=0).mean(dim=1)

    # print(std_summed.shape)
    # # Plot the stds as a histogram
    # plt.hist(std_summed.cpu().numpy(), bins=50)
    # plt.xlabel('Mean of stds')
    # plt.ylabel('Frequency')
    # plt.title('Mean of stds')
    # plt.savefig(f'sum_of_stds.png')
    # plt.close()

    print(class_prompts)
    return total_base_model_acc, total_plumber_acc, features, predictions

def learn_svm(val_features, val_labels, val_preds, test_features, test_labels, test_preds, args):

    # Number of unique values in val_preds
    print(f"Unique values in val_preds: {np.unique(val_preds)}")

    # Compare val_preds and val_gt and get the indices where they are equal
    val_correct = (val_preds == val_labels).astype(np.int)
    test_correct = (test_preds == test_labels).astype(np.int)

    print("Number of validation samples: ", len(val_correct))
    print("Number of test samples: ", len(test_correct))

    # # Number of unique values in val_correct
    # print(f"Unique values in val_correct: {np.unique(val_correct)}")
    print("Number of correct predictions in validation set: ", np.sum(val_correct))
    print("Number of correct predictions in test set: ", np.sum(test_correct))


    # Print Accuracy
    print(f"Validation Accuracy: {np.sum(val_correct)/len(val_correct)*100:.2f}%")
    print(f"Test Accuracy: {np.sum(test_correct)/len(test_correct)*100:.2f}%")

    # Scale the features
    scaler = StandardScaler()
    scaler.fit(val_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    if args.svm_checkpoint_path:
        svm_checkpoint_path = args.svm_checkpoint_path
    else:
        svm_checkpoint_path = os.path.join(args.save_dir, f'{args.svm_features}_single_svm_model.pkl')

    # Fit SVM is not saved
    if not os.path.exists(svm_checkpoint_path):
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.fit(val_features, val_correct)
    else:
        with open(svm_checkpoint_path, 'rb') as f:
            clf = pickle.load(f)

    test_correct_preds = clf.predict(test_features)

    test_accuracy = accuracy_score(test_correct,test_correct_preds)
    test_recall = recall_score(test_correct, test_correct_preds)
    test_precision = precision_score(test_correct, test_correct_preds)
    test_f1 = f1_score(test_correct, test_correct_preds)
    
    # Failure estimation report (class 1 represents correct prediction of task model)
    class_report = classification_report(test_correct, test_correct_preds, output_dict=True)

    cm = confusion_matrix(test_correct, test_correct_preds)
    class_level_acc = cm.diagonal()/cm.sum(axis=1)

    print("Number of failure and sucess samples in test", cm.sum(axis=1))

    estimated_accuracy = np.sum(test_correct_preds)/len(test_correct_preds) # Estimated accuracy from SVM for correct prediction
    task_model_accuracy = np.sum(test_correct)/len(test_correct) # Actuall prediction accuracy
    estimation_gap = np.abs(task_model_accuracy-estimated_accuracy)

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
    metrics_path = os.path.join(args.save_dir, f'{args.svm_features}_metrics_single_svm.json')
    if args.svm_checkpoint_path:
        metrics_path = os.path.join(args.save_dir, f'{args.svm_features}_metrics_single_svm_src.json')

    with open(metrics_path, 'w') as f:
        json.dump(out_metrics, f)

    # Print and log metrics
    metrics = f'{args.svm_features}\tAccuracy: {test_accuracy*100:.2f}%, Precision: {test_precision*100:.2f}%, Recall: {test_recall*100:.2f}%, F1 Score: {test_f1*100:.2f}%'
    
    print(metrics, "\n\n")
    with open(os.path.join(args.save_dir, 'single_svm_metrics.txt'), 'a') as f:
        f.write(metrics)
        f.write("\n")
    
    return clf

def get_features_preds(classifier, plumber, autoencoder, anchor_embeddings, clip_model, val_loader, test_loader, class_prompts):
    
    val_base_acc, val_plumber_acc, val_features, val_preds = evaluate(val_loader, classifier, plumber, autoencoder, anchor_embeddings, clip_model, class_prompts)
    test_base_acc, test_plumber_acc, test_features, test_preds = evaluate(test_loader, classifier, plumber, autoencoder, anchor_embeddings, clip_model, class_prompts)

    assert False
    return {
        # 'train': [train_features, train_preds],
        'val': [val_features, val_preds],
        'test': [test_features, test_preds]
    }

failure_caption_data_name_map = {
    'cifar10': 'CIFAR',
    'cifar10-c': 'CIFAR',
    'cifar10-limited': 'CIFAR',
    'CelebA': 'CELEBA',
    'cifar100': 'CIFAR100',
    'imagenet': 'IMAGENET',
}

def get_caption_scores_(plumber, captions, reference_caption, svm_fitter):
    caption_latent = plumber.encode_text_batch(captions)
    reference_latent = plumber.encode_text_batch([reference_caption])[0]
    latent = caption_latent - reference_latent

    out_mask = svm_fitter.predict(latent)
    decisions = svm_fitter.decision_function(latent)

    return decisions, caption_latent, out_mask

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

def main(args):
    
     # Load the CLIP model
    clip_model, clip_transform = clip.load(args.clip_model_name)
    clip_model = clip_model.to(args.device)

    classifier, train_transform, test_transform = build_classifier(args.classifier_name, num_classes=args.num_classes, 
                                                                    pretrained=args.use_imagenet_pretrained, 
                                                                    checkpoint_path=args.classifier_checkpoint_path)
    classifier = classifier.to(args.device)

    print(f"Built {args.classifier_name} classifier with checkpoint path: {args.classifier_checkpoint_path}")

    # Create the data loader and wrap them with Fabric
    train_dataset, val_dataset, test_dataset, failure_dataset, class_names = get_dataset(args.dataset_name, train_transform, test_transform, 
                                                            data_dir=args.data_dir, clip_transform=clip_transform, 
                                                            img_size=args.img_size, return_failure_set=True,
                                                            domain_name=args.domain_name, severity=args.severity, use_real=False)
    print(f"Using {args.dataset_name} dataset")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.train_on_testset, num_workers=8, pin_memory=True)
    failure_loader = torch.utils.data.DataLoader(failure_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

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
        # assert False, "{args.prompt_path} Prompt file not found."
        text_encodings = get_CLIP_text_encodings(clip_model, class_names, args.prompt_path)
        print(f"Saved CLIP {args.clip_model_name} text encodings to {args.prompt_path}")
    
    class_prompts = [f"This is a photo of a {class_name}" for class_name in class_names]

    plumber = PLUMBER(args.clip_model_name, args.num_classes, 
                      img_projection=args.img_projection, txt_projection=args.txt_projection, 
                      img_prompting=args.img_prompting, cls_txt_prompts=args.cls_txt_prompts, 
                      dataset_txt_prompt=args.dataset_txt_prompt, is_mlp=args.is_mlp, device=args.device)
    plumber = plumber.to(args.device)
    if args.step1_checkpoint_path and os.path.exists(args.step1_checkpoint_path):
        plumber.load_checkpoint(args.step1_checkpoint_path)
        print(f"Loaded PLUMBER from {args.step1_checkpoint_path}")
    else:
        print(f"PLUMBER checkpoint not found at {args.step1_checkpoint_path}")
    
    # Initialize the autoencoder
    
    autoencoder = AutoEncoder(input_dim=args.projection_dim, min_embed_dim=args.ae_dim, output_dim=args.projection_dim,
                               activation='relu', use_batch_norm=False, use_uq_wrapper=True)
    autoencoder = autoencoder.to(args.device)

    feature_type = 'proj' if 'proj' in args.svm_features else 'clip'

    # # Load the saved features and labels or create them
    features_path = os.path.join(args.save_dir, 'features', f'{feature_type}_features_labels_dict.pkl')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            features_pred_dict = pickle.load(f)
    else:
        # If the user passes the path to the autoencoder weights, load them: This is useful when dealing with multiple domains
        if args.ae_checkpoint_path:
            ae_checkpoint_path = args.ae_checkpoint_path
            if feature_type not in args.ae_checkpoint_path:
                assert False, f"Autoencoder checkpoint path {ae_checkpoint_path} does not match the feature type {feature_type}"
            anchor_checkpoint_path = os.path.join(os.path.dirname(ae_checkpoint_path), f'{feature_type}_anchor_embeddings.pt')
        else:
            # If autoencoder is available, load the weights
            ae_checkpoint_path = os.path.join(args.save_dir, f'{feature_type}_best_autoencoder_weights.pt')
            anchor_checkpoint_path = os.path.join(args.save_dir, f'{feature_type}_anchor_embeddings.pt')
        if os.path.exists(ae_checkpoint_path):
            autoencoder.load_state_dict(torch.load(ae_checkpoint_path))
            print(f"Loaded autoencoder from {ae_checkpoint_path}")
            anchor_embeddings = torch.load(anchor_checkpoint_path)
        else:
            # Define the loss function
            criterion = nn.L1Loss()
            # Create the optimizer and scheduler
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(autoencoder.parameters(), lr=args.learning_rate, momentum=0.9)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.learning_rate)

            start_epoch = 0
            best_loss = float('inf')
            train_losses = []
            # Train the autoencoder
            for epoch in range(start_epoch, args.num_epochs):
                
                train_loss, anchor_embeddings = train_one_epoch(
                                                    val_loader, plumber, autoencoder, optimizer,
                                                    class_prompts, text_encodings, criterion, epoch)
                
                train_losses.append(train_loss)

                # Plot the loss and save 
                plt.plot(train_losses)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Autoencoder Train Loss')
                plt.savefig(os.path.join(args.save_dir, f'{feature_type}_autoencoder_train_loss.png'))
                plt.close()

                if train_loss < best_loss:
                    best_loss = train_loss
                    print(f"Saving best autoencoder weights at epoch {epoch}")
                    torch.save(autoencoder.state_dict(), os.path.join(args.save_dir, f'{feature_type}_best_autoencoder_weights.pt'))
            #Save the autoencoder
            torch.save(autoencoder.state_dict(), os.path.join(args.save_dir, f'{feature_type}_autoencoder_final.pt'))

            # Save the anchor embeddings using pytorch
            torch.save(anchor_embeddings, os.path.join(args.save_dir, f'{feature_type}_anchor_embeddings.pt'))

        # Get the features and predictions
        features_pred_dict = get_features_preds(classifier, plumber, autoencoder, anchor_embeddings, clip_model, 
                                                failure_loader, test_loader, class_prompts)
        # Save the features and the failure labels
        os.makedirs(os.path.join(args.save_dir, 'features'), exist_ok=True)
        with open(features_path, 'wb') as f:
            pickle.dump(features_pred_dict, f)

    val_proj_features = features_pred_dict['val'][0][f'{feature_type}_features']
    val_ae_proj_features = features_pred_dict['val'][0][f'ae_{feature_type}_features']
    val_labels = np.asarray(features_pred_dict['val'][1]['gt_labels'])
    val_preds = np.asarray(features_pred_dict['val'][1]['classifier_preds'])

    test_proj_features = features_pred_dict['test'][0][f'{feature_type}_features']
    test_ae_proj_features = features_pred_dict['test'][0][f'ae_{feature_type}_features']
    test_labels = np.asarray(features_pred_dict['test'][1]['gt_labels'])
    test_preds = np.asarray(features_pred_dict['test'][1]['classifier_preds'])


    val_proj_features = np.stack(val_proj_features, axis=0)
    val_ae_proj_features = np.stack(val_ae_proj_features, axis=0)
    test_proj_features = np.stack(test_proj_features, axis=0)
    test_ae_proj_features = np.stack(test_ae_proj_features, axis=0)

    if 'ae_residual' in args.svm_features:
        val_features = val_proj_features - val_ae_proj_features
        test_features = test_proj_features - test_ae_proj_features
    else:
        val_features = val_ae_proj_features
        test_features = test_ae_proj_features

    svm_fitter = learn_svm(val_features, val_labels, val_preds, test_features, test_labels, test_preds, args)

    # Save the svm model
    with open(os.path.join(args.save_dir, f'{args.svm_features}_single_svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_fitter, f)

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

    parser.add_argument('--use_uq_wrapper', action='store_true', help='Whether to use uq_wrapper or not')
    parser.add_argument('--ae_dim', type=int, default=128, help='Dimension of the reduced autoencoder features')
    parser.add_argument('--svm_features', type=str, default='proj_ae', help='Features to use for SVM')
    parser.add_argument('--svm_checkpoint_path', type=str, help='Path to the SVM checkpoint')
    parser.add_argument('--ae_checkpoint_path', type=str, help='Path to the autoencoder checkpoint')

    args = parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    print(args)
    sys.stdout.flush()

    # Make directory for saving results
    args.save_dir = get_save_dir(args)    
    os.makedirs(args.save_dir, exist_ok=True)

    seed_everything(args.seed)

    # Save the args to a file
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    main(args)


"""

python train_auto_encoder.py \
    --data_dir './data/' \
    --domain_name "water" \
    --dataset_name "NICOpp" \
    --train_on_testset \
    --num_classes 60 \
    --batch_size 256 \
    --seed 42 \
    --img_size 224 \
    --img_prompt \
    --classifier_name "resnet18" \
    --classifier_checkpoint_path "logs/NICOpp/failure_estimation/autumn/resnet18/classifier/checkpoint_99.pth" \
    --step1_checkpoint_path "logs/NICOpp/failure_estimation/autumn/resnet18/plumber_img_prompt/_clsEpoch_99_bs_64_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/step_1/best_projector_weights.pth" \
    --clip_model_name "ViT-B/32" \
    --prompt_path "data/NICOpp/NICOpp_CLIP_ViT-B_32_text_embeddings.pth" \
    --n_promt_ctx 16 \
    --num_epochs 30 \
    --optimizer "sgd" \
    --learning_rate 0.1 \
    --val_freq 1 \
    --prefix "" \
    --proj_clip \
    --projection_dim 512 \
    --teacher_temp 2.0 \
    --student_temp 1 \
    --weight_img_loss 1.0 \
    --weight_txt_loss 1.0 \
    --num_gpus 1 \
    --num_nodes 1 \
    --use_uq_wrapper \
    --svm_features "clip_ae" \
    --ae_checkpoint_path logs/NICOpp/failure_estimation/autumn/resnet18/plumber_img_prompt/_clsEpoch_99_bs_64_lr_0.1_teT_2.0_sT_1.0_imgweight_1.0_txtweight_1.0_is_mlp_False/step_1/AE-failure_detector/autumn/uq_AE/clip_best_autoencoder_weights.pt


    """
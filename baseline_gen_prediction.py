#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Model Analysis (OOD Detection)

import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
import time

import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.modeldefs import resnet18, resnet50, LossEstimator
from models.wrn import WideResNet
from data.loader import get_loaders, get_cifar10c_loaders
import torchvision
import torchvision.transforms as tt
import matplotlib.pyplot as plt

from utils.util import *
from utils.cal_metric import *

def compute_msp(p):
    msp = torch.max(p, dim=1)[0]
    return msp

def compute_energy(logits, T=1.0):
    return -T*torch.logsumexp(logits/T, dim=1)

def entropy(p):
    """
    Function to compute predictive entropy
    :param p: Softmax probabilites of batch of samples from the classifier model - VGG

    :return
    :entropy: predictive_entropy of samples in a batch - (size : batch_size)
    """
    entropy = -torch.sum(p*torch.log(p + 1e-10), dim=1)
    return entropy

def choose_model(model_type, num_outputs, imagenet_ckpt_path=None):
    """
    Function that returns a torch model
    The function also changes the final layer of the chosen model to 512 x num_classes
    """
    """
    Function arguments: model_type (str), num_outputs (int), imagenet_ckpt_path (str)
    model type can be ['resnet18', 'resnet50', 'wrn']
    Returns : torch model
    """
    if model_type == 'resnet18':
        model = resnet18(imagenet_ckpt_path)
        model.linear = torch.nn.Linear(512, num_outputs)
        print('Model chosen: {}'.format(model_type))

    elif model_type == 'resnet50':
        model = resnet50(imagenet_ckpt_path)
        model.linear = torch.nn.Linear(512, num_outputs)
        print('Model chosen: {}'.format(model_type))

    elif model_type == 'wrn':
        model = WideResNet(num_classes=num_outputs, depth=40, widen_factor=2, dropRate=0.3)
        print('Model chosen: {}'.format(model_type))
    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Basic Analyses')
    parser.add_argument('--ckpt_dir', default='../ckpts', type=str, help='Ckpts directory')
    parser.add_argument('--model_type', default='wrn', type=str, help='Classifier model: (Choices:[resnet18, wrn])')
    parser.add_argument('--in_dataset_name', default='CIFAR10', type=str, help='Name of the In Dist. Dataset (CIFAR10, CIFAR100)')
    parser.add_argument('--ood_dataset_name', default='CIFAR10C', type=str, help='Name of the calibration dataset')
    parser.add_argument('--results_dir', default='../results', type=str, help='Results directory')
    parser.add_argument('--data_dir', default='/p/lustre1/viv41siv/projects/delta_uq_regression/data', type=str, help='Data directory')
    parser.add_argument('--score', default='energy', type=str, help='Baselines: (Choices:[msp, energy, pe, loss, scaled_loss])')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes (10, 100)')
    parser.add_argument('--train_type', default='vanilla', type=str, help='Classifier train type: (Choices:[vanilla, le])')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of dataloader workers')
    parser.add_argument('--recall_level', default=0.95, type=float, help='Threshold for TPR (OOD Detection)')
    parser.add_argument('--threshold', default=0.95, type=float, help='Threshold for OOD Detection/Gen.')

    parser.add_argument('--image_level', action='store_true', help='Image level gen. prediction with LE')
    parser.add_argument('--feature_level', action='store_true', help='Feature level gen. prediction with LE')
    parser.add_argument('--num_aug', default=10, type=int, help='Number of augmentations')

    # Arguments for CIFAR10C or CIFAR100C
    parser.add_argument('--corruption', default='gaussian_blur', type=str, help='CIFAR10C corruption')
    parser.add_argument('--severity', default=3, type=int, help='Severity of corruptions')
    args = parser.parse_args()
    return args

def compute_scores(loader, device, model, loss_est=None, softmax=None):
    """
    Function that iterates through the data loader and
    returns the args.score scores

    :return
    :args.score scores - (size: len(loader))
    """
    with torch.no_grad():
        # Lists to store logits, softmax probs, labels, loss
        logits_list, probs_list, loss_list, labels_list = [], [], [], []

        for i, (x, y) in enumerate(loader):
            ###############################################################
            x = x.to(device)
            y = y.to(device)

            if args.train_type == 'le':
                y_logits, features = model(x)
                loss = loss_est(features)
                loss_list.extend(loss.data.cpu().numpy())

            elif args.train_type == 'vanilla':
                y_logits, _ = model(x)

            logits_list.extend(y_logits.data.cpu().numpy())

            probs = softmax(y_logits)
            probs_list.extend(probs.data.cpu().numpy())

            labels_list.extend(y.data.cpu().numpy())
            print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(loader.dataset)))

        if args.score == 'msp':
            scores = compute_msp(torch.tensor(np.array(probs_list))).data.cpu().numpy()
        elif args.score == 'energy':
            scores = -compute_energy(torch.tensor(np.array(logits_list))).data.cpu().numpy()
        elif args.score == 'pe':
            scores = -predictive_entropy(torch.tensor(np.array(probs_list))).data.cpu().numpy()
        elif args.score == 'loss':
            scores = -np.array(loss_list).reshape(-1)
        elif args.score == 'scaled_msp':
            scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
            scores = compute_msp(softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
        elif args.score == 'scaled_pe':
            scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
            scores = -predictive_entropy(softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
        elif args.score == 'scaled_energy':
            T = torch.tensor(1.0+np.exp(np.array(loss_list)))
            logits = torch.tensor(np.array(logits_list))
            scores = T.view(-1)*torch.logsumexp(logits/T, dim=1)
            scores = scores.data.cpu().numpy()
        else:
            assert 'Not Implemented'

        return scores


class BaselineManager():
    """
    Class to manage baselines
    :param str model: name of classifier model
    :param DataLoader train_loader: iterate through labeled train data
    :param DataLoader val_loader: iterate through validation data
    :param dict config: dictionary with hyperparameters
    :return: object of TrainManager class

    Baseline 1: MSP
    Baseline 2: Energy
    Baseline 3: Predictive Entropy

    """
    def __init__(self, model=None, loss_est=None, in_loader=None, ood_loader=None, mean_std=None, device='cpu'):

        self.model = model
        self.loss_est = loss_est
        self.in_loader = in_loader
        self.ood_loader = ood_loader
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.mean_std = mean_std

        if args.train_type == 'le':
            self.loss_est.eval()
        self.model.eval()

    def perform_augmentation(self, x, aug=False):
        """
        NOTE: Here input x is a torch uint8 tensor [B, C, H, W]
        """
        # Only when aug. flag is selected
        if aug:
            # Following standard hyper-params
            trans_1 = tt.Compose([tt.RandAugment(num_ops=2, magnitude=15), tt.Resize((32,32))])
        else:
            trans_1 = tt.Compose([tt.Resize((32,32))])

        # trans_1 produces an output in the range [0, 255] in uint8. We convert that to float.  Before applying the CIFAR10 normalization, we rescale it [0, 1]
        x = trans_1(x).float()/255.0

        # We now normalize the data with CIFAR10 mean and std
        trans_2 = tt.Compose([tt.Normalize(*self.mean_std)])

        #Return the transformed images
        x = trans_2(x)
        return x


    def analyze_loss(self):
        all_losses = np.load(os.path.join(args.results_dir, args.model_type, args.in_dataset_name, 'losses.npy'))
        true_correctness = np.load(os.path.join(args.results_dir, args.model_type, args.in_dataset_name, 'true_correctness.npy'))
        print(all_losses.shape)
        print(len(true_correctness))
        print(np.mean(all_losses, 0))
        print(np.std(all_losses, 0))
        print('Max mean {}, Max std. {}'.format(np.mean(all_losses, 0).max(), np.std(all_losses, 0).max()))
        print('Min mean {}, Min std. {}'.format(np.mean(all_losses, 0).min(), np.std(all_losses, 0).min()))

        mean_losses = np.mean(all_losses, 0).reshape(-1)  #[:args.num_aug,:]
        std_losses = np.std(all_losses[:args.num_aug,:], 0).reshape(-1)  #[:args.num_aug,:]
        gen_error = ((-mean_losses <=1.86186).sum())/len(mean_losses)
        gen_accuracy = 1.0-gen_error
        print('Pred all', gen_accuracy)
        true_acc = sum(true_correctness)/len(true_correctness)
        print('True acc', true_acc)

        for i in range(args.num_aug+1):
            temp_l = all_losses[i,:].reshape(-1)
            gen_error = ((-temp_l <=1.86186).sum())/len(temp_l)
            gen_accuracy = 1.0-gen_error
            print('Pred only with actual', gen_accuracy)

        #gen_error = (((-all_losses[args.num_aug,:] <=2.3737373737373773) == (std_losses >=0.2)).sum())/len(std_losses)
        gen_error = (std_losses >=1.5).sum()/len(std_losses)
        gen_accuracy = 1.0-gen_error
        print('Hey', gen_accuracy)

        print(all_losses[:,0], np.std(all_losses[:args.num_aug,0]), true_correctness[0] )
        print(all_losses[:,4], np.std(all_losses[:args.num_aug,4]), true_correctness[4])
        print(all_losses[:,10], np.std(all_losses[:args.num_aug,10]), true_correctness[10])
        print(all_losses[:,1000], np.std(all_losses[:args.num_aug,1000]), true_correctness[1000])


    def msp_gen(self):
        """
        Computes the generalization error using the MSP metric
        """
        with torch.no_grad():
            msp_list, probs_list, labels_list = [], [], []
            correct, total = 0, 0
            correct_list = []

            for i, (x, y) in enumerate(self.ood_loader):
                ###############################################################
                x = self.perform_augmentation(x.clone(), False).to(self.device)
                y = y.to(self.device)
                y_logits, _ = self.model(x)
                probs = self.softmax(y_logits)
                msp_list.extend(compute_msp(probs).data.cpu().numpy())
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())

                _, predicted = y_logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                correct_list.append(predicted.eq(y).data.cpu().numpy())

                #print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.ood_loader.dataset)))

            msp_list = np.array(msp_list).reshape(-1)
            t = 0.68686868 #0.7474
            gen_error = ((msp_list <=t).sum())/len(msp_list)
            gen_accuracy = 1.0-gen_error
            print('\nMSP Based Gen. Prediction with t = 0.68686868')  #0.7474
            print('True Accuracy from Classifier {:.3f}'.format(correct/total))
            print('Predicted Accuracy', gen_accuracy)
            print(correct, total)


    def msp_gen_le(self):
        """
        Computes the generalization error using the MSP metric
        """
        with torch.no_grad():
            msp_list, probs_list, labels_list = [], [], []
            correct, total = 0, 0
            correct_list = []

            for i, (x, y) in enumerate(self.ood_loader):
                ###############################################################
                x = self.perform_augmentation(x.clone(), False).to(self.device)
                y = y.to(self.device)
                y_logits, features = self.model(x)
                loss = self.loss_est(features)
                scale = 1.0+torch.exp(loss)
                probs = self.softmax(y_logits/scale)
                msp_list.extend(compute_msp(probs).data.cpu().numpy())
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())

                _, predicted = y_logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                correct_list.append(predicted.eq(y).data.cpu().numpy())

                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.ood_loader.dataset)))

            msp_list = np.array(msp_list).reshape(-1)
            t = 0.676767676
            gen_error = ((msp_list <=t).sum())/len(msp_list)
            gen_accuracy = 1.0-gen_error
            print('\nMSP Based Gen. Prediction with t = 0.6262626')  #0.7474
            print('True Accuracy from Classifier {:.3f}'.format(correct/total))
            print('Predicted Accuracy', gen_accuracy)
            print(correct, total)

    def pseudo_label_aug_pred(self):
        # Load an image x, generate M randaug versions of x

        with torch.no_grad():
            # Lists to store logits, softmax probs, labels, loss
            probs_list, labels_list = [], []
            correct, total = 0, 0
            correct_list = []
            all_preds = np.array([])

            for i, (x, y) in enumerate(self.ood_loader):
                ###############################################################
                y = y.to(self.device)
                pred_label_list = []
                # Perform augmentation and obtain predictions
                for j in range(args.num_aug+1):   # Obtain predictions on x + all augmentations of x

                    if j == args.num_aug:
                        x_aug = self.perform_augmentation(x.clone(), False).to(self.device)
                    else:
                        x_aug = self.perform_augmentation(x.clone(), True).to(self.device)

                    y_logits, _ = self.model(x_aug)
                    probs = self.softmax(y_logits)
                    pred_label_list.extend(np.argmax(probs.data.cpu().numpy(),1).reshape(1,-1))

                    if j == args.num_aug:
                        _, predicted = y_logits.max(1)
                        total += y.size(0)
                        correct += predicted.eq(y).sum().item()
                        correct_list.append(predicted.eq(y).data.cpu().numpy())

                # Concatenate the predictions
                # np.array(pred_label_list).shape : args.num_aug+1 x no. of samples.
                all_preds = np.hstack([all_preds, np.array(pred_label_list)]) if all_preds.size else np.array(pred_label_list)
                #print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.ood_loader.dataset)))

            print(all_preds.shape)  # [11, 10000]
            print('Accuracy {:.3f}'.format(correct/total))

            ### Pseudo Label based Disagreement
            count = 5
            gen_accuracy = psedo_gen_pred(all_preds[:args.num_aug, :], count)
            print('\nPseudo Label based Disagreement with Count = {}'.format(count))
            print('True Accuracy from Classifier {:.3f}'.format(correct/total))
            print('Predicted Accuracy', gen_accuracy)


    def compute_thresholds(self):
        """
        Computes the threshold on the loss metric using the validation data
        """
        with torch.no_grad():
            score_list, probs_list, labels_list = [], [], []
            correct, total = 0, 0
            correct_list = []

            for i, (x, y) in enumerate(self.in_loader):
                ###############################################################
                x = x.to(self.device)
                y = y.to(self.device)
                y_logits, _ = self.model(x)
                probs = self.softmax(y_logits)
                score_list.extend(compute_msp(probs).data.cpu().numpy())
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())
                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.in_loader.dataset)))

            score_list = np.array(score_list).reshape(-1)
            err = np.argmax(np.array(probs_list), 1) != np.array(labels_list)
            thresholds = np.linspace(0,1,100)  # Possible thresholds
            max_loss = 10000
            for t in thresholds:
                 l = np.abs(np.mean((score_list<t)) - np.mean(err))  #np.abs(
                 print(l, t)
                 if l < max_loss:
                     max_loss = l
                     threshold = t

            print('Score type = {}'.format(args.score))
            print('Threshold for loss = {}'.format(threshold))  #0.7474
            gen_error = ((score_list <=threshold).sum())/len(score_list)
            gen_accuracy = 1.0-gen_error
            print(gen_accuracy)


    def compute_thresholds_le(self):
        """
        Computes the threshold on the loss metric using the validation data
        """
        with torch.no_grad():
            msp_list, probs_list, labels_list = [], [], []
            correct, total = 0, 0
            correct_list = []

            for i, (x, y) in enumerate(self.in_loader):
                ###############################################################
                x = x.to(self.device)
                y = y.to(self.device)
                y_logits, features = self.model(x)
                loss = self.loss_est(features)
                scale = 1.0+torch.exp(loss)
                probs = self.softmax(y_logits/scale)
                print(probs.shape)
                msp_list.extend(compute_msp(probs).data.cpu().numpy())
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())
                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.in_loader.dataset)))

            msp_list = np.array(msp_list).reshape(-1)
            err = np.argmax(np.array(probs_list), 1) != np.array(labels_list)
            thresholds = np.linspace(0,1,100)  # Possible thresholds
            max_loss = 10000
            for t in thresholds:
                 l = np.abs(np.mean((msp_list<t)) - np.mean(err))  #np.abs(
                 print(l, t)
                 if l < max_loss:
                     max_loss = l
                     threshold = t

            print('Threshold for MSP = {}'.format(threshold))  #0.7474
            gen_error = ((msp_list <=threshold).sum())/len(msp_list)
            gen_accuracy = 1.0-gen_error
            print(gen_accuracy)


    def compute_thresholds_all_gen(self):
        """
        Function that iterates through the data loader and
        returns the args.score scores

        :return
        :args.score scores - (size: len(loader))
        """
        with torch.no_grad():
            # Lists to store logits, softmax probs, labels, loss
            logits_list, probs_list, loss_list, labels_list = [], [], [], []

            for i, (x, y) in enumerate(self.in_loader):
                ###############################################################
                x = x.to(self.device)
                y = y.to(self.device)

                if args.train_type == 'le':
                    y_logits, features = self.model(x)
                    loss = self.loss_est(features)
                    loss_list.extend(loss.data.cpu().numpy())

                elif args.train_type == 'vanilla':
                    y_logits, _ = self.model(x)

                logits_list.extend(y_logits.data.cpu().numpy())
                probs = self.softmax(y_logits)
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())
                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.in_loader.dataset)))

            if args.score == 'msp':
                scores = compute_msp(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'energy':
                scores = -compute_energy(torch.tensor(np.array(logits_list))).data.cpu().numpy()
            elif args.score == 'pe':
                scores = -predictive_entropy(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'loss':
                scores = -np.array(loss_list).reshape(-1)
            elif args.score == 'scaled_msp':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = compute_msp(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_pe':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = -predictive_entropy(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_energy':
                T = torch.tensor(1.0+np.exp(np.array(loss_list)))
                logits = torch.tensor(np.array(logits_list))
                scores = T.view(-1)*torch.logsumexp(logits/T, dim=1)
                scores = scores.data.cpu().numpy()
            else:
                assert 'Not Implemented'

            scores = scores.reshape(-1)
            err = np.argmax(np.array(probs_list), 1) != np.array(labels_list)
            thresholds = np.linspace(-40, 40,5000)  # Possible thresholds
            max_loss = 10000
            for t in thresholds:
                 l = np.abs(np.mean((scores<t)) - np.mean(err))  #np.abs(
                 print(l, t)
                 if l < max_loss:
                     max_loss = l
                     threshold = t

            print('Model type = {}'.format(args.train_type))
            print('Threshold for {} = {}'.format(args.score, threshold))  #0.7474
            gen_error = ((scores <=threshold).sum())/len(scores)
            gen_accuracy = 1.0-gen_error
            print(gen_accuracy)

    def compute_thresholds_all_ood(self):
        """
        Function that iterates through the data loader and
        returns the args.score scores

        :return
        :args.score scores - (size: len(loader))
        """
        with torch.no_grad():
            # Lists to store logits, softmax probs, labels, loss
            logits_list, probs_list, loss_list, labels_list = [], [], [], []

            for i, (x, y) in enumerate(self.in_loader):
                ###############################################################
                x = x.to(self.device)
                y = y.to(self.device)

                if args.train_type == 'le':
                    y_logits, features = self.model(x)
                    loss = self.loss_est(features)
                    loss_list.extend(loss.data.cpu().numpy())

                elif args.train_type == 'vanilla':
                    y_logits, _ = self.model(x)

                logits_list.extend(y_logits.data.cpu().numpy())
                probs = self.softmax(y_logits)
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())
                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.in_loader.dataset)))

            if args.score == 'msp':
                scores = compute_msp(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'energy':
                scores = -compute_energy(torch.tensor(np.array(logits_list))).data.cpu().numpy()
            elif args.score == 'pe':
                scores = -predictive_entropy(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'loss':
                scores = -np.array(loss_list).reshape(-1)
            elif args.score == 'scaled_msp':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = compute_msp(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_pe':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = -predictive_entropy(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_energy':
                T = torch.tensor(1.0+np.exp(np.array(loss_list)))
                logits = torch.tensor(np.array(logits_list))
                scores = T.view(-1)*torch.logsumexp(logits/T, dim=1)
                scores = scores.data.cpu().numpy()
            else:
                assert 'Not Implemented'

            scores_id = scores.copy().reshape(-1)
            threshold = args.threshold
            # threshold = np.sort(scores_id)[::-1][int(args.recall_level*len(scores_id))]
            # print('Model type = {}'.format(args.train_type))
            # print('Threshold for {} OOD Detection = {}'.format(args.score, threshold))

            scores_ood = compute_scores(self.ood_loader, self.device, self.model, self.loss_est, self.softmax)
            true_id_labels = [1]*len(scores_id)
            true_ood_labels = [0]*len(scores_ood)
            pred_labels = []

            true_labels = np.concatenate((true_id_labels,true_ood_labels))
            all_scores = np.concatenate((scores_id,scores_ood))

            for s in all_scores:
                if s>=threshold:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            pred_labels = np.array(pred_labels)
            cm = confusion_matrix(true_labels, pred_labels)
            print(cm)
            print('\nIn Distribution = {}'.format(args.in_dataset_name))
            print('OOD = {}'.format(args.ood_dataset_name))
            print('Score = {}'.format(args.score))
            print('Train Type = {}'.format(args.train_type))
            print('Threshold for {} OOD Detection = {}'.format(args.score, threshold))  #0.7474
            print('FPR = {}'.format(cm[0][1]/(cm[0][1]+cm[0][0])))
            print('F1 Score = {}'.format(f1_score(true_labels, pred_labels)))

    def all_gen(self):
        """
        Function that iterates through the data loader and
        returns the args.score scores

        :return
        :args.score scores - (size: len(loader))
        """
        with torch.no_grad():
            # Lists to store logits, softmax probs, labels, loss
            logits_list, probs_list, loss_list, labels_list = [], [], [], []

            for i, (x, y) in enumerate(self.ood_loader):
                ###############################################################
                x = self.perform_augmentation(x.clone(), False).to(self.device)
                y = y.to(self.device)

                if args.train_type == 'le':
                    y_logits, features = self.model(x)
                    loss = self.loss_est(features)
                    loss_list.extend(loss.data.cpu().numpy())

                elif args.train_type == 'vanilla':
                    y_logits, _ = self.model(x)

                logits_list.extend(y_logits.data.cpu().numpy())
                probs = self.softmax(y_logits)
                probs_list.extend(probs.data.cpu().numpy())
                labels_list.extend(y.data.cpu().numpy())
                print('Processing {}/{} Images'.format((i+1)*x.shape[0], len(self.ood_loader.dataset)))

            if args.score == 'msp':
                scores = compute_msp(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'energy':
                scores = -compute_energy(torch.tensor(np.array(logits_list))).data.cpu().numpy()
            elif args.score == 'pe':
                scores = -predictive_entropy(torch.tensor(np.array(probs_list))).data.cpu().numpy()
            elif args.score == 'loss':
                scores = -np.array(loss_list).reshape(-1)
            elif args.score == 'scaled_msp':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = compute_msp(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_pe':
                scaled_logits = np.array(logits_list)/(1+np.exp(np.array(loss_list)))
                scores = -predictive_entropy(self.softmax(torch.tensor(np.array(scaled_logits)))).data.cpu().numpy()
            elif args.score == 'scaled_energy':
                T = torch.tensor(1.0+np.exp(np.array(loss_list)))
                logits = torch.tensor(np.array(logits_list))
                scores = T.view(-1)*torch.logsumexp(logits/T, dim=1)
                scores = scores.data.cpu().numpy()
            else:
                assert 'Not Implemented'

            scores = scores.reshape(-1)
            threshold = args.threshold
            true_acc = accuracy_score(np.array(labels_list), np.argmax(np.array(probs_list), 1))

            print('Model type = {}'.format(args.train_type))
            print('Corruption = {}, Severity = {}'.format(args.corruption, args.severity))
            print('Threshold for {} = {}'.format(args.score, threshold))  #0.7474
            gen_error = ((scores <=threshold).sum())/len(scores)
            gen_accuracy = 1.0-gen_error
            print('True Accuracy = {}'.format(true_acc))
            print('Predicted Accuracy = {}'.format(gen_accuracy))



def main():

    # Preliminary Checks
    if args.train_type == 'vanilla' and (args.score == 'loss' or args.score == 'scaled_msp' or args.score == 'scaled_pe' or args.score == 'scaled_energy'):
        assert False

    # Assigning the checkpoint names
    if args.train_type == 'vanilla':
        loss_est = None
        print(args.train_type)
        ckpt_name = 'ckpt_last.pth.tar'
    elif args.train_type == 'le':
        loss_est = LossEstimator()
        print(args.train_type)
        ckpt_name = 'ckpt_le.pth.tar'  #_last

    # Check if the checkpoint is existing or not
    #model_ckpt_path = os.path.join(args.ckpt_dir, args.model_type, args.in_dataset_name, ckpt_name)
    model_ckpt_path = os.path.join(args.ckpt_dir, args.model_type, 'cifar10', 'dan', 'le_epoch_99.pt')
    if not os.path.exists(model_ckpt_path):
        raise NotImplementedError
    else:
        print('Checkpoint exists')

    # Check and create results directory
    results_folder_path = os.path.join(args.results_dir, args.model_type,  args.in_dataset_name)
    check_and_create_path(results_folder_path)

    # Choosing the classifier
    model = choose_model(args.model_type, args.num_classes, imagenet_ckpt_path=None)

    # Assigning device (CUDA or CPU)
    device = cuda_or_cpu()

    # Loading the checkpoint
    if args.train_type == 'vanilla':
        model.load_state_dict(torch.load(model_ckpt_path)['state_dict'])
        print(model_ckpt_path)
        print('Loaded Model Checkpoint')
        #print(list(model.children()))
        model = model.to(device)
        model = multi_gpu(model)
        #print_network_parameters(model, args.model_type)

    elif args.train_type == 'le':
        #model.load_state_dict(torch.load(model_ckpt_path)['model_state_dict'])
        #loss_est.load_state_dict(torch.load(model_ckpt_path)['loss_est_state_dict'])
        print(model_ckpt_path)
        print('Loaded Model and LE Checkpoint')
        #print(list(model.children()))
        #print(list(loss_est.children()))
        model = model.to(device)
        model = multi_gpu(model)
        loss_est = loss_est.to(device)
        loss_est = multi_gpu(loss_est)
        model.load_state_dict(torch.load(model_ckpt_path)['model_state_dict'])
        loss_est.load_state_dict(torch.load(model_ckpt_path)['loss_est_state_dict'])
        #print_network_parameters(loss_est, 'LE')


    # Choose the data and the required transforms
    # Obtain the required transforms  (Other required transforms can be added as required)
    if args.in_dataset_name == 'CIFAR10':
        mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.in_dataset_name == 'CIFAR100':
        mean_std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    val_transform = tt.Compose([tt.ToTensor(), tt.Resize((32,32)), tt.Normalize(*mean_std)])
    #val_transform = tt.Compose([tt.PILToTensor()])
    # Obtaining In Dist. Val loader
    _, in_loader = get_loaders(args.data_dir, args.in_dataset_name, args.batch_size, val_transform, val_transform, args.num_workers)
    print('Loaded {} val loader'.format(args.in_dataset_name))

    # Obtaining OOD Loader
    if args.ood_dataset_name == 'CIFAR10C':
        data_dir = os.path.join(args.data_dir, 'CIFAR-10-c', 'CIFAR-10-C')
        val_transform = tt.Compose([tt.PILToTensor()])
        _, ood_loader = get_cifar10c_loaders(data_dir, args.batch_size, val_transform, args.severity, [args.corruption], args.num_workers)
    else:
        _, ood_loader = get_loaders(args.data_dir, args.ood_dataset_name, args.batch_size, val_transform, val_transform, args.num_workers)

    manager = BaselineManager(model, loss_est, in_loader, ood_loader, mean_std, device)

    #manager.analyze_loss()
    #manager.compute_thresholds()
    #manager.compute_thresholds_le()
    #manager.compute_thresholds_all_gen()
    manager.compute_thresholds_all_ood()
    #manager.all_gen()
    #manager.msp_gen()
    #manager.msp_gen_le()
    #manager.pseudo_label_aug_pred()


    #if args.image_level:
    #    manager.image_level_pred()
    #if args.feature_level:
    #    manager.feature_level_pred()
    #else:
    #    raise NotImplementedError


if __name__ == '__main__':
    start = time.time()
    args = parse_arguments()
    #corruption_list = ['brightness','fog','glass_blur','motion_blur','snow','contrast', 'frost','impulse_noise','pixelate','spatter','defocus_blur', 'gaussian_blur', 'jpeg_compression','saturate', 'speckle_noise','elastic_transform', 'gaussian_noise','shot_noise', 'zoom_blur']
    #corruption_list = ['shot_noise', 'zoom_blur']
    #for c in corruption_list:
    #    args.corruption = c
    main()
    end = time.time()
    time_elapsed = end - start
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # torchvision.utils.save_image(x, os.path.join(args.results_dir, args.model_type, args.in_dataset_name,'cifar10c.png'))

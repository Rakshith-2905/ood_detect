import numpy as np
import pandas as pd
import os
import json

import argparse
def read_json(file_path):
    data_all=[]
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        data_all.append(data)
    return data_all






def get_results(dataset_name, domain=None,baseline_path=None, pim_paths_dict=None):
    baseline_scores=['msp','energy','pe']
    pim_scores =['cross_entropy']

    all_results_domain=[]# this is for individual domain name
    for baseline_score in baseline_scores:
        
        baseline_file = os.path.join(baseline_path, baseline_score+'_results.json')
        
        baseline_data = read_json(baseline_file)

        domain_results_all = [f for f in baseline_data]# if f['domain_name']==domain]
        for domain_results in domain_results_all:
        
            results_to_append = [baseline_score, domain_results['true_test_acc'], domain_results['estimated_test_acc'],np.abs(domain_results['estimated_test_acc']-domain_results['true_test_acc']),
                                domain_results['test_failure_recall'],domain_results['test_success_recall'],domain_results['test_mathews_corr'], 
                                'None',
                                domain_results['train_domain_name'], domain_results['calib_domain_name'],domain_results['test_domain_name'],'None']
            all_results_domain.append(results_to_append)
    for pim_agg_method, pim_path in pim_paths_dict.items():
        for pim_score in pim_scores:
            pim_file = os.path.join(pim_path, f'{pim_score}_results.json')

            pim_data = read_json(pim_file)
            domain_data = pim_data
          
            domain_results_all = [f for f in domain_data]# if f['domain_name']==domain]
            for domain_results in domain_results_all:
                
                results_to_append = [pim_agg_method, domain_results['true_test_acc'], domain_results['estimated_test_acc'],np.abs( domain_results['estimated_test_acc']-domain_results['true_test_acc']),
                domain_results['test_failure_recall'],domain_results['test_success_recall'],\
                domain_results['test_mathews_corr'],
                domain_results['pim_model_test_acc'],
                domain_results['train_domain_name'], domain_results['calib_domain_name'],domain_results['test_domain_name'],
                domain_results['aggregator']]

                all_results_domain.append(results_to_append)
        
    #write all results_domain to a csv file
    df = pd.DataFrame(all_results_domain, columns=['method', 'true_test_acc', 'estimated_test_acc', 'gen gap', 'test_failure_recall', 'test_success_recall', 'test_mathews_corr','pim_model_test_acc','train_domain_name', 'calib_domain_name', 'test_domain_name','attribute_aggregation'])
    df.to_csv(f'{dataset_name}_all_results_all.csv', index=False)



   








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train desired classifier model on the desired Dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--domain', default= 'photo', type=str, help='Name of the domain if data is from DomainNet dataset')
    parser.add_argument('--baseline_path', type=str, help='Path to the baseline results')
    parser.add_argument('--pim_path', type=str, help='Path to the PIM results')
    parser.add_argument('--attribute_aggregation', default='mean', choices=['mha', 'mean', 'max'], help='Type of aggregation of the attribute scores')

    args = parser.parse_args()
    if args.dataset_name == 'pacs':
        baseline_path = 'logs/pacs-photo/resnet18/classifier/failure_results'
        pim_max_path = 'logs/pacs-photo/resnet18/mapper/_agg_max_bs_512_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.layer1/failure_results'
        pim_mean_path = 'logs/pacs-photo/resnet18/mapper/_agg_mean_bs_512_lr_0.001_augmix_prob_0.2_cutmix_prob_0.2_scheduler_warmup_epoch_0_layer_model.layer1/failure_results'
        pim_paths_dict= {'pim_max_path': pim_max_path}#, 'pim_mean_path': pim_mean_path}
        args.baseline_path = baseline_path
        args.pim_paths_dict = pim_paths_dict

    #get_results(args.dataset_name, args.domain, args.baseline_path, args.pim_paths_dict)
    for domain in ['photo', 'art_painting', 'cartoon', 'sketch']:
        get_results(args.dataset_name, args.domain, args.baseline_path, args.pim_paths_dict)


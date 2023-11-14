import os
import argparse
import datetime
import json
import pandas as pd
import torch

import utils
import similarity


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/32", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="resnet50", 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
# parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
#                     help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
#                           Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--target_layers", type=str, default="layer",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="broden", 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden", 'custom_domainnet'])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()
def read_words_from_file(filepath):
    with open(filepath, 'r') as f:
        return f.read().split('\n')


def read_label_names(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file]


def save_dataframe_to_csv(df, directory, target_model):
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    save_path = os.path.join(directory, f"{target_model}_{datetime.datetime.now().strftime('%y_%m_%d_%H_%M')}")
    os.mkdir(save_path)
    df_path = os.path.join(save_path, "descriptions.csv")
    df.to_csv(df_path, index=False)
    return save_path


def main(args):
    args.target_layers = args.target_layers.split(",")
    similarity_fn = eval(f"similarity.{args.similarity_fn}")
    
    utils.save_activations(clip_name=args.clip_model, target_name=args.target_model, target_layers=args.target_layers,
                           d_probe=args.d_probe, concept_set=args.concept_set, batch_size=args.batch_size, device=args.device,
                           pool_mode=args.pool_mode, save_dir=args.activation_dir)
    
    words = read_words_from_file(args.concept_set)
    label_names = read_label_names('../ood_detect/data/domainnet_v1.0/class_names.txt')

    outputs = {"layer": [], "unit": [], "description": [], "similarity": []}
    outputs_2 = {"label_name": [], "layer":[], "unit": [], "description": [], "similarity": []}

    for target_layer in args.target_layers:
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names

        similarities, target_feats = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True, device=args.device
        )
        vals, ids = torch.max(similarities, dim=1)
        
        del similarities
        torch.cuda.empty_cache()
        
        descriptions = [words[int(idx)] for idx in ids]
        
        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer]*len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())

        selected_image_index = 5
        top_k = 5
        # Extracting activations for the selected image
        selected_image_activations = target_feats[selected_image_index]

        # Finding top K activations
        top_k_values, top_k_indices = torch.topk(selected_image_activations, top_k)

        # Fetch descriptions for the top K activations
        descriptions = [words[int(idx)] for idx in top_k_indices]
        
        print(f"Top {top_k} activations for image index {selected_image_index}: {descriptions}")


        # for neuron_idx in top_k_neurons:
        #     similarity_value, concept_idx = vals[neuron_idx].max(0)
        #     outputs_2["layer"].append(target_layer)
        #     outputs_2["unit"].append(neuron_idx.item())
        #     outputs_2["description"].append(words[int(concept_idx)])
        #     outputs_2["similarity"].append(similarity_value.item())

        # for label_id, neuron_indices in top_k_neurons_per_label.items():
        #     for neuron_idx in neuron_indices:
        #         similarity_value, concept_idx = similarities[neuron_idx].max(0)
        #         # outputs_2["label_name"].append(label_names[int(label_id)])
        #         outputs_2["layer"].append(target_layer)
        #         outputs_2["unit"].append(neuron_idx.item())
        #         outputs_2["description"].append(words[int(concept_idx)])
        #         outputs_2["similarity"].append(similarity_value.item())

    save_path = save_dataframe_to_csv(pd.DataFrame(outputs), args.result_dir, args.target_model)
    
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    df_2 = pd.DataFrame(outputs_2)
    df_2.to_csv(os.path.join(save_path, "descriptions_2.csv"), index=False)


def class_top_k_activations(data, labels, K=5):
    # Identify unique labels
    unique_labels = torch.unique(labels)
    
    top_k_indices_per_label = {}
    
    for label in unique_labels:
        # Filter out the rows corresponding to the current label
        label_data = data[labels == label]
        
        # Compute the average along the num_images axis
        avg_values = torch.mean(label_data, dim=0)
        
        # Extract the top K values and their indices
        _, top_k_indices = torch.topk(avg_values, K)
        
        top_k_indices_per_label[label.item()] = top_k_indices

    return top_k_indices_per_label

def top_k_neurons_data(activations, K=5):
    # Compute the average along the num_images axis
    avg_values = torch.mean(activations, dim=0)
    # Extract the top K values and their indices
    _, top_k_indices = torch.topk(avg_values, K)

    return top_k_indices


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
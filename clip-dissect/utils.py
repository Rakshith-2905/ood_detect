import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader



PM_SUFFIX = {"max":"_max", "avg":""}

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    all_labels = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        if isinstance(dataset, DataLoader):
            for _, labels, images in tqdm(dataset):
                features = model.encode_images(images.to(device))
                all_features.append(features)
                all_labels.append(labels.cpu())
        else:
            for _, labels, images in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                features = model.encode_images(images.to(device))
                all_features.append(features)
                all_labels.append(labels.cpu())

    torch.save(torch.cat(all_features), save_name)
    torch.save(torch.cat(all_labels), save_name.replace(".pt", "_labels.pt"))
    print(f"Saved image features to {save_name}")
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)

    print(f"Saved text features to {save_name}")
    del text_features
    torch.cuda.empty_cache()
    return

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """


    
    # If target layers is not a list, make it a list
    if not isinstance(target_layers, list):
        target_layers = [target_layers]

    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    hooks = {}
    for target_layer in target_layers:
        if target_layer == "projector":
            command = "target_model.projector.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        elif target_layer == "proj":
            pass
        else:
            command = "target_model.clip_model.visual.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
        print(f"Saving activations from {command}")
        
    with torch.no_grad():
        clip_outputs = []
        if isinstance(dataset, DataLoader):
            for _, labels, images in tqdm(dataset):
                features = target_model.encode_images(images.to(device))
                if "proj" in target_layers:
                    clip_outputs.append(features)
        else:
            for _, labels, images in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
                features = target_model.encode_images(images.to(device))
                if "proj" in target_layers:
                    clip_outputs.append(features)
    
    for target_layer in target_layers:
        if target_layer == "proj":
            torch.save(torch.cat(clip_outputs), save_names[target_layer])
        else:
            torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
        print(f"Saved {target_layer} activations to {save_names[target_layer]}")
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_activations(clip_model, target_layers, d_probe, 
                     concept_set, batch_size, pool_mode, save_names, device):
    
    #ignore empty lines
    concept_set = [i for i in concept_set if i!=""]
    
    # text = clip.tokenize(["{}".format(word) for word in concept_set]).to(device)
    
    target_save_name, clip_save_name, text_save_name = save_names

    save_clip_text_features(clip_model, concept_set, text_save_name, batch_size)
    save_clip_image_features(clip_model, d_probe, clip_save_name, batch_size, device)
    save_target_activations(clip_model, d_probe, target_save_name, target_layers,
                            batch_size, device, pool_mode)
     
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda"):
    
    similarity_fn = eval(f"{similarity_fn}")
    
    image_features = torch.load(clip_save_name, map_location='cpu').float()
    text_features = torch.load(text_save_name, map_location='cpu').float()

    print(f"Image feature matrix shape: {image_features.shape}")
    print(f"Text feature matrix shape: {text_features.shape}")
    # Compute similarity matrix between all images and all concepts
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        clip_similarity_matrix = 100*(image_features @ text_features.T) # (num_images, num_concepts)

    del image_features, text_features
    torch.cuda.empty_cache()
   
    target_activations = torch.load(target_save_name, map_location='cpu').float() # (num_images, num_target_activations)
    print(f"Target feature matrix shape: {target_activations.shape}")

    clip_similarity = similarity_fn(clip_similarity_matrix, target_activations, device=device) # (num_target_activations, num_concepts)
    print(f"Target Activation-Concept Similarity matrix (CLIP) shape: {clip_similarity.shape}")

    torch.cuda.empty_cache()

    if return_target_feats:
        return clip_similarity, target_activations, clip_similarity_matrix
    else:
        del target_activations 
        torch.cuda.empty_cache()
        return clip_similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def cos_similarity_cubed(clip_feats, target_feats, device='cuda', batch_size=10000, min_norm=1e-3):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
        target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
        
        clip_feats = clip_feats**3
        target_feats = target_feats**3
        
        clip_feats = clip_feats/torch.clip(torch.norm(clip_feats, p=2, dim=0, keepdim=True), min_norm)
        target_feats = target_feats/torch.clip(torch.norm(target_feats, p=2, dim=0, keepdim=True), min_norm)
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def cos_similarity(clip_feats, target_feats, device='cuda'):
    with torch.no_grad():
        clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        
        batch_size = 10000
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device))
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)

def soft_wpmi(clip_feats, target_feats, top_k=100, a=10, lam=1, device='cuda',
                        min_prob=1e-7, p_start=0.998, p_end=0.97):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        p_in_examples = p_start-(torch.arange(start=0, end=top_k)/top_k*(p_start-p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm(range(target_feats.shape[1])):
            
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            
            curr_p_d_given_e = 1+p_in_examples*(curr_clip_feats-1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
            torch.cuda.empty_cache()

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) - 
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def wpmi(clip_feats, target_feats, top_k=28, a=2, lam=0.6, device='cuda', min_prob=1e-7):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        for orig_id in tqdm(range(target_feats.shape[1])):
            torch.cuda.empty_cache()
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            curr_p_d_given_e = torch.sum(torch.log(curr_clip_feats+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) -
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))

        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

def rank_reorder(clip_feats, target_feats, device="cuda", p=3, top_fraction=0.05, scale_p=0.5):
    """
    top fraction: percentage of mostly highly activating target images to use for eval. Between 0 and 1
    """
    with torch.no_grad():
        batch = 1500
        errors = []
        top_n = int(target_feats.shape[0]*top_fraction)
        target_feats, inds = torch.topk(target_feats, dim=0, k=top_n)

        for orig_id in tqdm(range(target_feats.shape[1])):
            clip_indices = clip_feats.gather(0, inds[:, orig_id:orig_id+1].expand([-1,clip_feats.shape[1]])).to(device)
            #calculate the average probability score of the top neurons for each caption
            avg_clip = torch.mean(clip_indices, dim=0, keepdim=True)
            clip_indices = torch.argsort(clip_indices, dim=0)
            clip_indices = torch.argsort(clip_indices, dim=0)
            curr_errors = []
            target = target_feats[:, orig_id:orig_id+1].to(device)
            sorted_target = torch.flip(target, dims=[0])

            baseline_diff = sorted_target - torch.cat([sorted_target[torch.randperm(len(sorted_target))] for _ in range(5)], dim=1)
            baseline_diff = torch.mean(torch.abs(baseline_diff)**p)
            torch.cuda.empty_cache()

            for i in range(math.ceil(clip_indices.shape[1]/batch)):

                clip_id = (clip_indices[:, i*batch:(i+1)*batch])
                reorg = sorted_target.expand(-1, batch).gather(dim=0, index=clip_id)
                diff = (target-reorg)
                curr_errors.append(torch.mean(torch.abs(diff)**p, dim=0, keepdim=True)/baseline_diff)
            errors.append(torch.cat(curr_errors, dim=1)/(avg_clip)**scale_p)

        errors = torch.cat(errors, dim=0)
    return -errors

 
import torch
from torch.nn.functional import one_hot
from torch.nn import functional as F
from tqdm import tqdm
import random


def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    dist_matrix = torch.cat(dist_matrix, dim=1)
    return dist_matrix

def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten() # same memory
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1
    return mapping_matrix

def resize(images, size=(224, 224)):
    return F.interpolate(images, size=size, mode='bilinear', align_corners=True)


def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]
    return modified_logits

def generate_label_mapping_by_frequency(resnet_model, projector, compute_similarities, data_loader, mapping_num = 1, similarity_mode='cosine', text_encodings=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(resnet_model, "eval"):
        resnet_model.eval()
    if hasattr(projector, "eval"):
        projector.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    with torch.no_grad():
        for resnet_logits, resnet_embeddings, labels in pbar:

            resnet_embeddings = resnet_embeddings.to(device)
            probs_from_resnet = F.softmax(resnet_logits, dim=-1)
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)           
            # Compute similarities between image embeddings and text encodings
            fx0 = compute_similarities(proj_embeddings, text_encodings, mode=similarity_mode)

            fx0s.append(fx0.cpu().float())
            ys.append(labels.cpu().int())


            # Dele the variables to free up memory
            del resnet_embeddings, resnet_logits, probs_from_resnet, proj_embeddings, fx0

            if len(fx0s) == 1500:
                break
    # Randomly select only half of fx0s and ys
    # num_to_select = int(len(fx0) * 0.30)

    # # Zip the lists together, shuffle the combined list, and take the required number of pairs
    # combined = list(zip(fx0, ys))
    # random.shuffle(combined)
    # selected_pairs = combined[:num_to_select]
    # # Unzip the selected pairs back into two lists
    # fx0, ys = zip(*selected_pairs)

    fx0s = torch.cat(fx0s).cpu().float()
    ys = torch.cat(ys).cpu().int()
    if ys.size(0) != fx0s.size(0):
        assert fx0s.size(0) % ys.size(0) == 0
        ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
    dist_matrix = get_dist_matrix(fx0s, ys)
    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    print(len(mapping_sequence))
    return mapping_sequence

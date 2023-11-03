import torch
from torch.nn.functional import one_hot
from torch.nn import functional as F
from tqdm import tqdm
import random


def get_dist_matrix_chunk(fx, y, num_classes):
    dist_matrix_chunk = torch.zeros((num_classes, num_classes), device=fx.device)
    fx = one_hot(torch.argmax(fx, dim=-1), num_classes=num_classes)
    for i in range(num_classes):
        dist_matrix_chunk[:, i] += fx[y == i].sum(0)
    return dist_matrix_chunk

def update_dist_matrix(dist_matrix, dist_matrix_chunk):
    dist_matrix += dist_matrix_chunk

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
    resnet_model.eval()
    projector.eval()
        
    num_classes = text_encodings.size(0)
    dist_matrix = torch.zeros((num_classes, num_classes), device="cpu")
    fx0s_chunk = []
    ys_chunk = []
    batch_counter = 0

    chunk_size = 1000

    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    with torch.no_grad():
        for resnet_logits, resnet_embeddings, labels,_ in pbar:
            labels = labels.to(device)
            resnet_embeddings = resnet_embeddings.to(device)
            probs_from_resnet = F.softmax(resnet_logits, dim=-1)
            # Project the resnet embeddings
            proj_embeddings = projector(resnet_embeddings)           
            # Compute similarities between image embeddings and text encodings
            fx0 = compute_similarities(proj_embeddings, text_encodings, mode=similarity_mode)

            
            fx0s_chunk.append(fx0.cpu().float())
            ys_chunk.append(labels.cpu().int())
            batch_counter += 1

            if batch_counter >= chunk_size:
                # Update the dist_matrix with the accumulated chunk
                fx0s_chunk = torch.cat(fx0s_chunk)
                ys_chunk = torch.cat(ys_chunk)
                dist_matrix_chunk = get_dist_matrix_chunk(fx0s_chunk, ys_chunk, num_classes)
                update_dist_matrix(dist_matrix, dist_matrix_chunk)

                # Clear the accumulated chunk from memory
                del fx0s_chunk, ys_chunk, dist_matrix_chunk
                torch.cuda.empty_cache()  # Clear cache if using CUDA
                fx0s_chunk = []
                ys_chunk = []
                batch_counter = 0

    # Handle the last chunk if there is any data left
    if fx0s_chunk:
        fx0s_chunk = torch.cat(fx0s_chunk)
        ys_chunk = torch.cat(ys_chunk)
        dist_matrix_chunk = get_dist_matrix_chunk(fx0s_chunk, ys_chunk, num_classes)
        update_dist_matrix(dist_matrix, dist_matrix_chunk)
        del fx0s_chunk, ys_chunk, dist_matrix_chunk
    print(dist_matrix.shape)

    pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
    mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
    print(len(mapping_sequence))

    assert False

    resnet_model.train()
    projector.train()

    return mapping_sequence

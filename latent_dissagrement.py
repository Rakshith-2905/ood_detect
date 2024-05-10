import numpy as np
import faiss
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

def get_knn_index(all_features):
    index = faiss.IndexFlatL2(all_features.shape[1])  # L2 distance for similarity
    index.add(all_features)
    return index

def get_knn_mat(anchor_features, test_features, n_neighbours, knn_index=None):
    """
    The function compares each feature vector in the search_features set to all 
    feature vectors in the anchor_features set to find their nearest neighbors within the dataset.
    Args:
    - anchor_features: list of anchor features which are used to find nearest neighbours for search features
    - test_features: list of features for which nearest neighbours are to be found in the anchor features
    - n_neighbours: number of neighbours to consider for similarity metrics
    - knn_index: faiss index for kNN search
    Returns:
    - indices: indices of the nearest neighbours in the anchor features for each search feature

    """

    if knn_index is None:
        # Create kNN index which is used to search for nearest neighbours of the search features
        knn_index = faiss.IndexFlatL2(anchor_features.shape[1])
        knn_index.add(np.ascontiguousarray(anchor_features))
    
    # Search for nearest neighbours using kNN (k = n_neighbours + 1 to exclude self from neighbours)
    D, indices = knn_index.search(np.ascontiguousarray(test_features), n_neighbours + 1)
    is_include_same = np.array_equal(test_features, anchor_features[indices[:, 0]])
    if is_include_same:
        return indices[:, 1:]
    return indices[:, :n_neighbours]

def get_latent_disaggrement(n_neighbours, test_features, anchor_features, metric='auroc', verbose=False, knn_index=None):
    """
    The function computes the evaluation scores for the given metric using the test and anchor features.
    This computes GT neighbours for the test features in the anchor features 
    Args:
    - n_neighbours: number of neighbours to consider for similarity metrics
    - test_features: list of features for which nearest neighbours are to be found in the anchor features, 
                    the first element if foundation model features and the remaining are test features from model to be evaluated
    - anchor_features: list of anchor features which are used to find nearest neighbours for search features, 
                    the first element if foundation model features and the remaining are test features from model to be evaluated
    - metric: metric to use for evaluation
    - verbose: whether to show progress bar
    - knn_index: faiss index for kNN search
    Returns:
    - scores: evaluation scores for the given metric

    """

    # Compute the GT nearest neighbours for the test features in the anchor features
    # This is computed using the test and anchor features from a reliable encoder like a Foundation model
    main_test_features, main_anchor_features = test_features[0], anchor_features[0]

    # Normalize features
    normed_main_test_features = main_test_features / np.linalg.norm(main_test_features, axis=1)[:, np.newaxis]
    normed_main_anchor_features = main_anchor_features / np.linalg.norm(main_anchor_features, axis=1)[:, np.newaxis]
    
    gt_neighbor_indices = get_knn_mat(normed_main_anchor_features, normed_main_test_features, n_neighbours)
    
    gt_neighbor_mask = np.zeros((len(main_test_features), len(main_anchor_features)))
    for i in range(len(gt_neighbor_indices)):
        gt_neighbor_mask[i][gt_neighbor_indices[i]] = 1

    # Compute cosine similarity between the test features and the anchor features
    main_cos_sim = cosine_similarity(main_test_features, main_anchor_features)
    
    # For every other test
    all_auroc_list = []
    for j in range(1, len(anchor_features)):
        other_test_features, other_anchor_features = test_features[j], anchor_features[j]
        
        normed_other_test_features = other_test_features / np.linalg.norm(other_test_features, axis=1)[:, np.newaxis]
        normed_other_anchor_features = other_anchor_features / np.linalg.norm(other_anchor_features, axis=1)[:, np.newaxis]

        other_neighbor_indices = get_knn_mat(normed_other_anchor_features, normed_other_test_features, n_neighbours)
        
        other_neighbor_mask = np.zeros((len(normed_other_test_features), len(normed_other_anchor_features)), dtype=bool)
        for i in range(len(other_neighbor_indices)):
            other_neighbor_mask[i][other_neighbor_indices[i]] = 1
        
        cos_sim = cosine_similarity(normed_other_test_features, normed_other_anchor_features)

        auroc_list = []
        range_obj = tqdm(range(len(cos_sim))) if verbose else range(len(cos_sim))
        for i in range_obj:
            if metric == 'ndcg_rank':
                auroc_list.append(ndcg_score([gt_neighbor_mask[i]], [cos_sim[i]], k=n_neighbours))

            elif metric == 'jaccard':
                intersection = np.logical_and(gt_neighbor_mask[i], other_neighbor_mask[i]).sum()
                union = np.logical_or(gt_neighbor_mask[i], other_neighbor_mask[i]).sum()
                auroc_list.append(intersection / union)

            elif metric == 'spearmanr':
                rank_coef, _ = spearmanr(main_cos_sim[i], cos_sim[i])
                auroc_list.append(rank_coef)

            else:
                raise ValueError(f"Invalid metric: {metric}")
        
        print(f"Mean {metric} for test {j}: {np.mean(auroc_list)} with len: {len(auroc_list)}")
        all_auroc_list.append(auroc_list)
    all_auroc_list = np.array(all_auroc_list)

    return all_auroc_list.mean(axis=0)

# Sample evaluation script
if __name__ == "__main__":
    num_anchors = 5000
    num_test_samples = 5000
    feature_dim = 512
    n_neighbours = 5

    # Generate random features
    test_features = [np.random.rand(num_test_samples, feature_dim) for _ in range(2)]
    anchor_features = [np.random.rand(num_anchors, feature_dim) for _ in range(2)]

    search_mask = np.random.choice([False, True], size=num_test_samples)

    # Evaluate using different metrics
    for metric in ['ndcg_rank', 'jaccard', 'spearmanr']:
        scores = get_latent_disggrement(n_neighbours, test_features, anchor_features, metric=metric, verbose=True)
        print(f"Evaluation scores for {metric}:", scores)

        assert False

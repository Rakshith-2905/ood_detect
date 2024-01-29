import torch

class ClusterCreater:
    def __init__(self, num_classes, samples_per_class, embedding_dim):
        self.data_dict = {i: torch.empty((0, embedding_dim)).to('cuda') for i in range(num_classes)}
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.embedding_dim = embedding_dim
        self.mean_var_dict = {}

    def update_embedding(self, class_id, embedding):
        if self.data_dict[class_id].shape[0] < self.samples_per_class:
            self.data_dict[class_id] = torch.cat((self.data_dict[class_id], embedding.detach().unsqueeze(0)), 0)
        else:
            self.data_dict[class_id] = torch.cat((self.data_dict[class_id][1:],
                                                     embedding.detach().view(1, -1)), 0)

    def update_embeddings_batch(self, labels, embeddings):
        for class_id, embedding in zip(labels, embeddings):
            labels = class_id.to('cpu').item()
            embedding = embedding
            self.update_embedding(labels, embedding)
        
        self.calculate_mean_variance()

    def calculate_mean_variance(self):
        X, mean_embed_id = None, None
        for index in range(self.num_classes):
            if index == 0:
                X = self.data_dict[index] - self.data_dict[index].mean(0)
                mean_embed_id = self.data_dict[index].mean(0).unsqueeze(0)
            else:
                X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                mean_embed_id = torch.cat((mean_embed_id, self.data_dict[index].mean(0).unsqueeze(0)), 0)

        eye_matrix = torch.eye(self.embedding_dim).to('cuda')
        temp_precision = torch.mm(X.T, X) / len(X)
        temp_precision += 0.0001 * eye_matrix

        self.mean_var_dict = {
            'mean': mean_embed_id,
            'covariance': temp_precision
        }

    def sample_embeddings(self, num_samples, select, class_specific=False):
        ood_samples = []
        for index in range(self.num_classes):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                self.mean_var_dict['mean'][index], covariance_matrix=self.mean_var_dict['covariance'])
            negative_samples = new_dis.rsample((num_samples*100,))
            prob_density = new_dis.log_prob(negative_samples)
            
            cur_samples, index_prob = torch.topk(-prob_density, select)
            ood_samples.append(negative_samples[index_prob])

        if class_specific:
            return ood_samples
        # Combine samples from all classes
        ood_samples = torch.cat(ood_samples, 0)

        return ood_samples
    
    def sample_embeddings_custom(self, num_samples, segment, class_specific=False):

        # Note:
        # segment = 0 -> selects the top 20% of samples with the highest probability densities.
        # segment = 4 -> selects the bottom 20% of samples with the lowest probability densities.
        
        assert 0 <= segment < 5, "Segment must be in the range [0, 4]."

        ood_samples = []
        for index in range(self.num_classes):
            # Create the distribution
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                self.mean_var_dict['mean'][index], covariance_matrix=self.mean_var_dict['covariance'])

            # Sample and compute log probabilities
            negative_samples = new_dis.rsample((num_samples * 100,))
            log_prob_density = new_dis.log_prob(negative_samples)

            # Sort log probabilities in descending order and split into 5 segments
            sorted_indices = torch.argsort(log_prob_density, descending=True)
            segment_size = len(sorted_indices) // 5
            segment_start = segment_size * segment
            segment_end = segment_start + segment_size if segment < 4 else len(sorted_indices)

            # Get the indices for the desired segment and select corresponding samples
            index_prob = sorted_indices[segment_start:segment_end]
            selected_samples = negative_samples[index_prob]

            
            selected_samples = selected_samples[:num_samples] if len(selected_samples) > num_samples else selected_samples

            ood_samples.append(selected_samples)

        if class_specific:
            return ood_samples
        else:
            # Combine samples from all classes
            combined_samples = torch.cat(ood_samples, 0)
            return combined_samples

    def save_mean_covariance(self, file_path):
        torch.save(self.mean_var_dict, file_path)


if __name__ == '__main__':
    # Instantiate the EmbeddingManager
    manager = EmbeddingManager(num_classes=10, samples_per_class=100, embedding_dim=128)

    # Generate and update random embeddings for each class
    for class_id in range(10):
        for _ in range(100):
            random_embedding = torch.rand(128)  # Random embedding
            manager.update_embedding(class_id, random_embedding)

    # Calculate mean and variance
    manager.calculate_mean_variance()

    print("Mean:", manager.mean_var_dict['mean'].shape
          , "Covariance:", manager.mean_var_dict['covariance'].shape)


    # # Optional: Sample some embeddings (e.g., 5 samples for selection, selecting top 3)
    # sampled_embeddings = manager.sample_embeddings(num_samples=5, select=3)
    # print("Sampled Embeddings:", sampled_embeddings)

    # Sample embeddings and check for outlierness
    manager.sample_embeddings(num_samples=5, select=3)
    for class_id in range(10):
        mean = manager.mean_var_dict['mean'][class_id]
        std_dev = torch.sqrt(torch.diag(manager.mean_var_dict['covariance'])).reshape(-1, 128)

        sampled_embeddings = manager.sample_embeddings(num_samples=100, select=5)
        distances = torch.norm(sampled_embeddings - mean, dim=1)

        # Check if distances are generally larger than a few standard deviations
        outlier_condition = distances > (2 * std_dev).mean()  # Example threshold
        print(f"Class {class_id} - Are sampled embeddings outliers? {outlier_condition.all()}")
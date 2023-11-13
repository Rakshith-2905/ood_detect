import torch
from torch.utils.data import Dataset
import os
import glob

class ChunkedDataset(Dataset):
    def __init__(self, save_path, feature_extractor_name):
        self.save_path = save_path
        self.feature_extractor_name = feature_extractor_name
        self.chunk_files = sorted(glob.glob(f"{self.save_path}/{self.feature_extractor_name}_*.pt"))
        
        # Precompute and store the start and end indices for each chunk
        self.chunk_indices = []
        self.cumulative_index = 0  # Cumulative index to handle non-sequential chunks
        for chunk_file in self.chunk_files:
            parts = os.path.basename(chunk_file).split('_')
            start_index, end_index = int(parts[-2]), int(parts[-1].split('.')[0])
            self.chunk_indices.append((self.cumulative_index, self.cumulative_index + (end_index - start_index)))
            self.cumulative_index += (end_index - start_index + 1)  # Update cumulative index

        self.current_chunk_data = None
        self.current_chunk_start = -1
        self.current_chunk_end = -1

        # Total size is the cumulative index at the end
        self.total_size = self.cumulative_index

    def _load_chunk(self, chunk_file, start_index, end_index):
        self.current_chunk_data = torch.load(chunk_file)
        self.current_chunk_start = start_index
        self.current_chunk_end = end_index

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Check if idx is within the range of the currently loaded chunk
        if not (self.current_chunk_start <= idx <= self.current_chunk_end):
            # Find the chunk that contains idx and load it
            for chunk_file, (start, end) in zip(self.chunk_files, self.chunk_indices):
                if start <= idx <= end:
                    self._load_chunk(chunk_file, start, end)
                    break

        within_chunk_index = idx - self.current_chunk_start
        image_features = self.current_chunk_data['image_features'][within_chunk_index]
        text_features = self.current_chunk_data['text_features'][within_chunk_index]
        return image_features, text_features


if __name__ == '__main__':
    save_path = '/p/gpfs1/KDML/feats/train'
    feature_extractor_name = 'dino_vits16'

    dataset = ChunkedDataset(save_path, feature_extractor_name)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    for image_features, text_features in data_loader:
        # print(image_features.shape)
        # print(text_features.shape)
        pass
        # Process your data here
        

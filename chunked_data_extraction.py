import torch
from torch.utils.data import Dataset
import os
import glob
import pandas as pd


class ChunkedDataset(Dataset):
    def __init__(self, save_path, feature_extractor_name):
        self.save_path = save_path
        self.feature_extractor_name = feature_extractor_name
        self.chunk_files = sorted(glob.glob(f"{self.save_path}/{self.feature_extractor_name}_*.pt"))
        
        # Precompute and store the start and end indices for each chunk
        self.chunk_indices = []
        self.cumulative_index = 0  # Cumulative index to handle non-sequential chunks
        error_files=[]
        for chunk_file in self.chunk_files:
            # try:
            #     data = torch.load(chunk_file)
            # except:
            #     error_files.append(chunk_file)
                
            #     continue
            parts = os.path.basename(chunk_file).split('_')
            start_index, end_index = int(parts[-2]), int(parts[-1].split('.')[0])
            self.chunk_indices.append((self.cumulative_index, self.cumulative_index + (end_index - start_index)))
            self.cumulative_index += (end_index - start_index + 1)  # Update cumulative index

        self.current_chunk_data = None
        self.current_chunk_start = -1
        self.current_chunk_end = -1

        # Total size is the cumulative index at the end
        self.total_size = self.cumulative_index
    #     # add error files to error_files.txt
    #     # for each error_file use pandas
    #     start_indices=[]
    #     end_indices=[]
    #     for error_file in error_files:
    #         # get the start and end indices
    #         parts = os.path.basename(error_file).split('_')
    #         start_index, end_index = int(parts[-2]), int(parts[-1].split('.')[0])
    #         # add the start and end indices to error_files.txt
    #         start_indices.append(start_index)
    #         end_indices.append(end_index+1)
    #     data = {'start_indices': start_indices, 'end_indices': end_indices}
    #     df = pd.DataFrame(data, columns=['start_indices', 'end_indices'], index=None)
    #   #  df.to_csv('error_files.txt', sep='\t', index=False)

    #     assert False

        
    def _load_chunk(self, chunk_file, start_index, end_index):

        self.current_chunk_data = torch.load(chunk_file,map_location="cpu")
        self.current_chunk_start = start_index
        self.current_chunk_end = end_index
       # print(f"Loaded chunk {chunk_file.split('/')[-1]} with indices {start_index} to {end_index},length {len(self.current_chunk_data['image_features'])} ")

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

def group_and_save_chunks(original_path, grouped_path, feature_extractor_name, group_size):
    chunk_files = sorted(glob.glob(f"{original_path}/{feature_extractor_name}_*.pt"))
    current_group = {'image_features': [], 'text_features': []}
    group_index = 0  # Index to keep track of the group number

    num_images = 0
    for chunk_file in chunk_files:
        data = torch.load(chunk_file)
        current_group['image_features'].extend(data['image_features'])
        current_group['text_features'].extend(data['text_features'])

        num_images += data['image_features'].shape[0]
        print(num_images)
        # Check if the current group reached the desired group size
        if num_images >= group_size:
            start_index = group_index * group_size
            end_index = start_index + num_images - 1

            # Save the current group
            save_filename = f"{grouped_path}/{feature_extractor_name}_{start_index}_{end_index}.pt"
            torch.save(current_group, save_filename)
            print(f"Saved grouped chunk: {save_filename}")

            # Prepare for the next group
            current_group = {'image_features': [], 'text_features': []}
            group_index += 1
            num_images = 0

    # Save the last group if it's not empty
    if current_group['image_features']:
        start_index = group_index * group_size
        end_index = start_index + len(current_group['image_features']) - 1
        save_filename = f"{grouped_path}/{feature_extractor_name}_{start_index}_{end_index}.pt"
        torch.save(current_group, save_filename)
        print(f"Saved grouped chunk: {save_filename}")


if __name__ == '__main__':
    save_path = '/p/gpfs1/KDML/feats/test'
    feature_extractor_name = 'dino_vits16'

    # dataset = ChunkedDataset(save_path, feature_extractor_name)
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # for image_features, text_features in data_loader:
    #     # print(image_features.shape)
    #     # print(text_features.shape)
    #     pass
    #     # Process your data here
        
    # Usage
    original_path = '/p/gpfs1/KDML/feats/test'
    grouped_path = '/p/gpfs1/KDML/feats/test_grouped'
    feature_extractor_name = 'dino_vits16'
    group_size = 1000000  # Define the size of each group based on your needs

    group_and_save_chunks(original_path, grouped_path, feature_extractor_name, group_size)
        

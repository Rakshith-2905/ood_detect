import os
import pandas as pd
import numpy as np

def create_chunks(json_file, mode):
    # Read the JSON file
    data_json = pd.read_json(json_file, orient='records', lines=True)

    # Get the length of the data
    data_len = len(data_json)

    # Set chunk size and range per job
    chunk_size = 20000
    range_per_job = 100000

    # Create chunks using numpy
    indices = np.arange(0, data_len, range_per_job)
    data_chunks_start_indices=[]
    data_chunks_end_indices=[]

    for i in range(len(indices) - 1):
        start_index = indices[i]

        end_index = indices[i + 1] - 1
        data_chunks_start_indices.append(start_index)
        data_chunks_end_indices.append(end_index)
        print(f"{start_index}-{end_index}")

    # Handle the last chunk separately to include the remaining elements
    if indices[-1] < data_len:
        print(f"{indices[-1]}-{data_len - 1}")
        data_chunks_start_indices.append(indices[-1])
        data_chunks_end_indices.append(data_len - 1)

    # Print the first 10 start and end indices
    print(data_chunks_start_indices[:10], data_chunks_end_indices[:10])

    # Create a dictionary with start and end indices
    data = {'start_indices': data_chunks_start_indices, 'end_indices': data_chunks_end_indices}

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['start_indices', 'end_indices'], index=None)

    # Define the output file path based on the mode
    output_file = f"{mode}_chunks.txt"

    # Save the DataFrame to a text file
    df.to_csv(output_file, sep='\t', index=False)

# Example usage:
json_file_path = "/usr/workspace/KDML/yfcc/yfcc15m_clean_open_data_already_downloaded_test_final.json"
mode = "test"

create_chunks(json_file_path, mode)

json_file_path = "/usr/workspace/KDML/yfcc/yfcc15m_clean_open_data_already_downloaded_train_final.json"
mode = "train"
create_chunks(json_file_path, mode)  
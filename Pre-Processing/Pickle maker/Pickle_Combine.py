import os
import pickle
import gzip
import numpy as np
from numba import cuda
from tqdm import tqdm

def load_pickle(file_path):
    with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path), 'rb') as f:
        data = pickle.load(f)
        print("\n" + file_path + " loaded.")
    return data

def save_pickle(data, file_path):
    with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path), 'wb') as f:
        pickle.dump(data, f)
        print("\n" + file_path + " saved.")

@cuda.jit
def mark_unique(names1, names2, mask, len_names1):
    i = cuda.grid(1)
    if i < names2.size:
        unique = True
        for j in range(len_names1):
            if names2[i] == names1[j]:
                unique = False
                break
        mask[i] = unique

def merge_pickles(file1_path, file2_path, output_path):
    # Load both pickle files
    data1 = load_pickle(file1_path)
    data2 = load_pickle(file2_path)
    
    # Determine the smaller file to minimize comparisons
    if len(data1) > len(data2):
        data1, data2 = data2, data1
    
    # Convert 'name' fields to NumPy arrays
    names1 = np.array([entry['name'] for entry in data1])
    names2 = np.array([entry['name'] for entry in data2])
    
    # Allocate memory on the GPU for the arrays
    d_names1 = cuda.to_device(names1)
    d_names2 = cuda.to_device(names2)
    d_mask = cuda.device_array(len(names2), dtype=np.bool_)
    
    # Define the number of threads and blocks
    threads_per_block = 256
    blocks_per_grid = (len(names2) + (threads_per_block - 1)) // threads_per_block
    
    # Show progress of checking each entry using tqdm
    progress_bar = tqdm(total=blocks_per_grid, desc=f"Merging {file1_path} with {file2_path}")
    
    # Call the GPU function
    mark_unique[blocks_per_grid, threads_per_block](d_names1, d_names2, d_mask, len(names1))
    
    # Update progress bar
    progress_bar.update(blocks_per_grid)
    progress_bar.close()
    
    # Copy the result back to the host
    mask = d_mask.copy_to_host()
    
    # Extract unique entries from data2
    unique_data2 = [data2[i] for i in range(len(data2)) if mask[i]]
    
    # Merge the data
    merged_data = data1 + unique_data2
    
    # Save the merged data back to a new pickle file
    save_pickle(merged_data, output_path)

# Define a function to merge multiple pickle files iteratively
def merge_multiple_pickles(file_paths, output_path):
    if len(file_paths) < 2:
        raise ValueError("At least two pickle files are required to merge.")
    
    temp_output = 'temp_merged.pkl.gz'
    
    # Use tqdm to show progress of the overall merging process
    overall_progress_bar = tqdm(total=len(file_paths) - 1, desc="Overall Merging Progress")
    
    for i in range(1, len(file_paths)):
        if i == 1:
            merge_pickles(file_paths[0], file_paths[i], temp_output)
        else:
            merge_pickles(temp_output, file_paths[i], temp_output)
        overall_progress_bar.update(1)
    
    overall_progress_bar.close()
    
    # Save the final merged output
    os.rename(temp_output, output_path)
    print(output_path + " saved.")

# List of pickle file paths
file_paths = [
    'Dataset\\Pickles\\train_checkpoint_0.pkl',
    'Dataset\\Pickles\\train_checkpoint_1.pkl',
    'Dataset\\Pickles\\train_checkpoint_2.pkl',
    'Dataset\\Pickles\\train_checkpoint_3.pkl',
    'Dataset\\Pickles\\train_checkpoint_4.pkl',
    'Dataset\\Pickles\\train_checkpoint_5.pkl',
    'Dataset\\Pickles\\train_checkpoint_6.pkl',
    'Dataset\\Pickles\\train_checkpoint_7.pkl',
    'Dataset\\Pickles\\train_checkpoint_8.pkl',
    'Dataset\\Pickles\\train_checkpoint_9.pkl'
]

# Final output path
final_output_path = 'Dataset\\Pickles\\excel_data.train'

# Merge all pickle files
merge_multiple_pickles(file_paths, final_output_path)

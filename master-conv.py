import os
import h5py
import numpy as np
from tqdm import tqdm


def find_fullres_folders(top_folder):
    fullres_paths = []
    for root, dirs, files in os.walk(top_folder):
        if 'FullRes' in dirs:
            fullres_paths.append(os.path.join(root, 'FullRes\\'))
    return fullres_paths


def process_h5(input_file, output_file, chunk_size=10):
    with h5py.File(input_file, 'r') as infile:
        dataset = infile['video_data' \
                         '']
        t, x, y, z, c = dataset.shape
        times = np.arange(1, t + 1)

        with h5py.File(output_file, 'w') as outfile:
            # Create datasets with uint8 dtype
            data_dset = outfile.create_dataset('data', (t, c, z, y, x), dtype=np.uint8)
            times_dset = outfile.create_dataset('times', (t,), dtype=times.dtype)

            # Write "times" data
            times_dset[:] = times

            # Write "data" in chunks
            for i in tqdm(range(0, t, chunk_size), desc=f"Processing {input_file}...", leave=False):
                end = min(i + chunk_size, t)
                chunk_data = dataset[i:end] // 256  # Bit-wise right shift
                chunk_data = chunk_data.astype(np.uint8)
                chunk_data = np.transpose(chunk_data, (0, 4, 3, 2, 1))
                data_dset[i:end] = chunk_data


top_folder = "E:\\scape-data\\finished-nwb\\"
all_folders = find_fullres_folders(top_folder)

# Sort folders by the size of their video-array.h5 files
all_folders.sort(key=lambda folder: os.path.getsize(f"{folder}video-array.h5"))

for k in tqdm(range(len(all_folders)), desc="Overall"):
    each_folder = all_folders[k]
    if each_folder == 'E:\\scape-data\\finished-nwb\\nwb-error-sample-20231017-worm0run0\\FullRes\\':
       process_h5(f'{each_folder}video-array.h5', f'{each_folder}uint8_data.h5')
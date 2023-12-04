import h5py
import nd2reader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random

def update_frame(val):
    """
    Update the displayed frame based on the slider's position.
    """
    frame_index = int(slider.val)
    frame = dataset[frame_index, :, :]
    img.set_data(frame)
    fig.canvas.draw_idle()

def visualize_frames_with_slider(file_path, dataset_name):
    """
    Visualizes frames from a specified dataset in an HDF5 file with a slider to browse through them.
    """
    global dataset, img, slider, fig

    with h5py.File(file_path, 'r') as hdf:
        if dataset_name in hdf:
            dataset = hdf[dataset_name]

            # Initial frame
            frame_index = 0
            initial_frame = dataset[frame_index, :, :]

            # Setting up the plot
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.1, bottom=0.25)
            img = ax.imshow(initial_frame, cmap='gray')
            ax.set_title(f'Frame {frame_index} from {dataset_name}')

            # Slider
            ax_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
            slider = Slider(ax_slider, 'Frame', 0, dataset.shape[0] - 1, valinit=frame_index, valfmt='%0.0f')

            # Update the frame when the slider is moved
            slider.on_changed(update_frame)

            plt.show()
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")

def extract_metadata(nd2_file_path):
    """
    Extracts metadata from a .nd2 file.

    :param nd2_file_path: Path to the .nd2 file
    :return: Dictionary containing metadata
    """
    with nd2reader.ND2Reader(nd2_file_path) as images:
        metadata = images.metadata
    return metadata

def print_metadata(obj, prefix=''):
    """
    Recursively prints the metadata of the HDF5 object (group/dataset).
    """
    print(f"{prefix}{obj.name}")
    for key, val in obj.attrs.items():
        print(f"{prefix}  Metadata: {key} = {val}")

    if isinstance(obj, h5py.Group):
        for key in obj:
            print_metadata(obj[key], prefix + '  ')

# Example usage of the function
nd2_file_path = 'C:\\Users\\Kevin\\Documents\\data\\2022-06-14-03.nd2'  # Replace with the path to your .nd2 file
print(nd2reader.ND2Reader(nd2_file_path))
print(nd2reader.ND2Reader(nd2_file_path).metadata['num_frames'])
with nd2reader.ND2Reader(nd2_file_path) as images:
    num_channels = images.sizes['c']
    for channel in range(num_channels):
        # Accessing each channel
        channel_image = images.get_frame_2D(c=channel, t=500)

        # Display or process the channel image here
        # For demonstration, let's just show the image using matplotlib
        plt.imshow(channel_image, cmap='gray')
        plt.title(f'Channel {channel}')
        plt.show()

# Replace 'your_large_file.h5' with the path to your large HDF5 file
file_path = 'C:\\Users\\Kevin\\Documents\\data\\2022-06-14-01.h5'
with h5py.File(file_path, 'r') as hdf:
    dataset = 'img_nir'
    if dataset in hdf:
        data = hdf[dataset]
        print(data)
        print(data.shape)
    else:
        print(f"Dataset {dataset} not found in the file.")

# Visualize frames with a slider
visualize_frames_with_slider(file_path, 'img_nir')

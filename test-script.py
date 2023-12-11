import h5py
import nd2reader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
import nrrd

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

def nd2_test(nd2_file_path):
    print(nd2reader.ND2Reader(nd2_file_path))
    print(nd2reader.ND2Reader(nd2_file_path).metadata['num_frames'])
    with nd2reader.ND2Reader(nd2_file_path) as images:
        x_size = images.sizes['x']
        y_size = images.sizes['y']
        z_size = nd2reader.ND2Reader(nd2_file_path).metadata['z_coordinates'][0]
        num_channels = images.sizes['c']
        for channel in range(num_channels):
            # Accessing each channel
            channel_image = images.get_frame_2D(c=channel, t=500)
            plt.imshow(channel_image, cmap='gray')
            plt.title(f'Channel {channel}')
            plt.show()

def h5_test(file_path):
    with h5py.File(file_path, 'r') as hdf:
        for dataset in hdf:
            print(dataset)
        dataset = 'pos_feature'
        if dataset in hdf:
            data = hdf[dataset][:]
            print(data.shape)
            print(data[5])
            print(data[25])
            print(data[55])
            print(data[75])
            print(data[-115])
        else:
            print(f"Dataset {dataset} not found in the file.")

    # Visualize frames with a slider
    visualize_frames_with_slider(file_path, 'img_nir')

def extract_second_channel(nd2_path, frame_index):
    """
    Extract the second channel from a specified frame in an .nd2 file.
    """
    with nd2reader.ND2Reader(nd2_path) as images:
        return images.get_frame_2D(c=1, t=frame_index)

def create_rgb_image(paths, frame_index):
    """
    Create an RGB image from the second channels of a specified frame from three .nd2 files.
    """
    channels = [extract_second_channel(path, frame_index) for path in paths]
    rgb_image = np.stack(channels, axis=-1)
    rgb_image = (255 * (rgb_image / rgb_image.max())).astype(np.uint8)
    return rgb_image

def update(val):
    """
    Update the image based on the slider's position.
    """
    frame_index = int(slider.val)
    rgb_image = create_rgb_image(nd2_paths, frame_index)
    ax.imshow(rgb_image)
    fig.canvas.draw_idle()

def test_nrrd(file_path):
    """
    Reads a .nrrd file and prints its metadata.

    Parameters:
    file_path (str): The path to the .nrrd file.
    """
    try:
        file, header = nrrd.read(file_path, index_order='C')

        print(header['sizes'])
        print(header['space directions'])
        print(np.shape(np.array(file)))

    except Exception as e:
        print(f"An error occurred: {e}")


"""
nd2_paths = ['C:\\Users\\sep27\\Documents\\[X] Data\\flavell\\2022-06-14-03.nd2', 'C:\\Users\\sep27\\Documents\\[X] Data\\flavell\\2022-06-14-04.nd2', 'C:\\Users\\sep27\\Documents\\[X] Data\\flavell\\2022-06-14-05.nd2']


# Determine the number of frames in the smallest file
min_frames = min(nd2reader.ND2Reader(f).sizes['t'] for f in nd2_paths)

# Create the initial RGB image
frame_index = 0
rgb_image = create_rgb_image(nd2_paths, frame_index)

# Create the plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.imshow(rgb_image)

# Create the slider
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = Slider(ax_slider, 'Frame', 0, min_frames-1, valinit=frame_index, valfmt='%0.0f')

# Update the image when the slider value changes
slider.on_changed(update)

plt.show()
"""

file_path = 'C:\\Users\\sep27\\Documents\\[X] Data\\flavell\\2022-06-14-01.h5'
h5_test(file_path)

## Replace 'path_to_file.nrrd' with the actual file path
#test_nrrd('C:\\Users\\sep27\\Documents\\[X] Data\\flavell\\proj_neuropal\\2022-06-14-01\\neuron_rois.nrrd')

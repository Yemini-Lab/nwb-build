import scipy.io
import configparser
import datetime
import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict
import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import skimage.io as skio
from dateutil import tz
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, \
    MultiChannelVolume, MultiChannelVolumeSeries
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position
from pynwb.ophys import ImageSegmentation, PlaneSegmentation, \
    DfOverF, RoiResponseSeries
from tifffile import TiffFile
from tifffile import imread
from tqdm import tqdm

def iterate_folders():
    cwd = os.getcwd()
    existing_nwb_files = {f[:-4] for f in os.listdir(cwd) if f.endswith('.nwb')}

    for item in os.listdir(cwd):
        if os.path.isdir(item) and item not in existing_nwb_files:
            print(f"\nProcessing: {item}")
            dt, subject_number, run_number = extract_info(item)
            identifier = f"{dt}/{subject_number}/{run_number}"
            print(f"-Date: {dt}\n-Subject #: {subject_number}\n-Run #: {run_number}\n-Identifier: {identifier}")

            if all(x != "Unknown" for x in [dt, subject_number, run_number]):
                metadata = {
                    'dt': dt,
                    'subject_number': subject_number,
                    'run_number': run_number,
                    'identifier': identifier
                }
                full_path = os.path.join(cwd, item)
                metadata = parse_metadata(full_path, metadata)
                metadata = parse_ini(full_path, metadata)
                build_nwb(full_path, metadata)
                print(f"Saved {item}.")
            elif all(x != "Unknown" for x in [dt, subject_number]) and run_number == 'Unknown':
                metadata = {
                    'dt': dt,
                    'subject_number': subject_number,
                    'run_number': 'reverse',
                    'identifier': identifier
                }
                full_path = os.path.join(cwd, item)
                metadata = parse_metadata(full_path, metadata)
                metadata = parse_ini(full_path, metadata)
                build_nwb(full_path, metadata)
                print(f"Saved {item}.")


def extract_info(folder_name):
    # Extract datetime
    dt_match = re.search(r'\d{8}', folder_name)
    dt = dt_match.group(0) if dt_match else "Unknown"

    # Extract worm number
    worm_match = re.search(r'worm(\d+)', folder_name, re.IGNORECASE)
    subject_number = f"worm-{worm_match.group(1)}" if worm_match else "Unknown"

    # Extract run number
    run_match = re.search(r'run(\d+)', folder_name, re.IGNORECASE)
    run_number = f"run-{run_match.group(1)}" if run_match else "Unknown"

    return dt, subject_number, run_number


def convert_mat_to_python(mat_array: Any) -> Any:
    """Convert MATLAB arrays to Python-native types."""
    if np.isscalar(mat_array):
        return mat_array.item()
    elif mat_array.size == 1:
        return mat_array.item().decode('utf-8') if isinstance(mat_array.item(), bytes) else mat_array.item()
    else:
        return mat_array.tolist()


def parse_metadata(full_path: str, metadata: Dict[str, Any]) -> None:
    # Load MATLAB info
    mat_file = next(f for f in os.listdir(full_path) if f.endswith('_info.mat'))
    mat_data = scipy.io.loadmat(os.path.join(full_path, mat_file))
    info = mat_data['info'][0, 0]

    # Populate metadata
    metadata['camera'] = {field: convert_mat_to_python(info['camera'][field][0, 0]) for field in
                          info['camera'].dtype.names}
    metadata['daq'] = {field: convert_mat_to_python(info['daq'][field][0, 0]) for field in info['daq'].dtype.names}

    # Handle non-serializable types in 'aiLabels'
    if 'aiLabels' in metadata.get('daq', {}):
        metadata['daq']['aiLabels'] = [
            elem.decode('utf-8') if isinstance(elem, bytes) else elem.tolist() if isinstance(elem, np.ndarray) else elem
            for elem in metadata['daq']['aiLabels']]

    # Update date format
    scan_start_time = info['scanStartTimeApprox'][0]
    formatted_time = datetime.strptime(scan_start_time, '%d-%b-%Y %H:%M:%S').strftime('%Y%m%d %H:%M:%S')
    if metadata['dt'] in formatted_time:
        metadata['dt'] = formatted_time

    # Add comments from ReadMe.txt, if exists
    readme_path = os.path.join(full_path, 'ReadMe.txt')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            metadata['comments'] = f.read()

    # Pretty print metadata
    try:
        print("-Metadata:")
        print(json.dumps(metadata, indent=4))
    except TypeError as e:
        print(f"Error: {e}")

    return metadata


def parse_ini(full_path, metadata):
    # Initialize configparser object
    config = configparser.ConfigParser()

    # Read nwb.ini from the folder
    ini_path = os.path.join(full_path, 'nwb.ini')
    config.read(ini_path)

    # Update identifier if needed
    if not config['SESSION']['identifier']:
        config['SESSION']['identifier'] = metadata['identifier']
    else:
        metadata['identifier'] = config['SESSION']['identifier']

    # Update metadata dictionary
    for section in config.sections():
        metadata[section.lower()] = {}
        for key, value in config[section].items():
            metadata[section.lower()][key] = value

    # Write changes back to nwb.ini
    with open(ini_path, 'w') as configfile:
        config.write(configfile)

    return metadata


# Discover and sort tiff files
# Read .nd2 file and extract frames
def discover_nd2_files(file_path):
    nd2_file = nd2reader.ND2Reader(file_path)
    frames = len(nd2_file)
    channels = nd2_file.metadata['channels']
    print(f"-FRAME COUNT: {frames}")
    print(f"-CHANNEL COUNT: {len(channels)}")
    return nd2_file, frames, channels


def h5_memory_mapper(nd2_file, output_file):
    if os.path.exists(output_file):
        print(f"-HDF5 FILE ALREADY EXISTS, SKIPPING BUILD.")
        return

    shape = nd2_file.sizes
    h5_file = h5py.File(output_file, 'w')
    h5_file.create_dataset('dataset', shape, dtype='uint16')

    try:
        print("Populating the .h5 file...")
        for i in tqdm(range(shape[0]), desc="Processing frames"):
            frame_data = nd2_file.get_frame_2D(c=i)
            h5_file['dataset'][i, :, :, :, :] = frame_data
    finally:
        print("Flushing changes and closing the file.")
        h5_file.close()


def iter_calc_h5(filename, numZ):
    with h5py.File(filename, 'r') as h5_file:
        t, x, y, z, c = h5_file['dataset'].shape

        for i in tqdm(range(t), desc="Processing time points"):
            tpoint = np.zeros((x, y, z, c), dtype='uint16')
            for j in range(numZ):
                tpoint[:, :, j, :] = h5_file['dataset'][i, :, :, j, :]

            yield np.squeeze(tpoint)


# DETECT & BUILD MIP GROUP
def build_mip(nwbfile, ImagingVol, full_path):
    video_path = f"{full_path}/MIP.mp4"
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert to grayscale; modify as needed
    cap.release()

    mip_data = np.array(frames)  # Shape should be (T, X, Y)
    mip_data = mip_data[:, np.newaxis, :, :, np.newaxis]  # Reshape to (T, X, Y, 1, 1) to fit (T, X, Y, Z, C)
    hefty_data = H5DataIO(data=mip_data, compression=True)

    mip_vol_series = MultiChannelVolumeSeries(
        name='MaximumIntensityProjection',
        description='Maximum Intensity Projection video.',
        data=hefty_data,
        unit='',
        timestamps=list(range(hefty_data.shape[0])),
        imaging_volume=ImagingVol  # Assuming this is defined earlier
    )

    # we create a processing module for the pre-processed neuroPAL image
    mip_module = nwbfile.create_processing_module(
        name='MaxIntensityProj',
        description='Maximum Intensity Projection'
    )
    mip_module.add(mip_vol_series)

    return mip_vol_series


def build_gcamp(nwbfile, full_path, OptChannels, OpticalChannelRefs, device, metadata):

    nd2_file, frames, channels = discover_nd2_files(os.path.join(full_path))
    h5_memory_mapper(nd2_file, os.path.join(full_path, 'FullRes', 'video-array.h5'))
    scan_rate = metadata['scan_rate']
    ai_sampling_rate = metadata['ai_sampling_rate']

    numZ = nd2_file.sizes['z']
    numX = nd2_file.sizes['x']
    numY = nd2_file.sizes['y']

    # Create DataChunkIterator
    data = DataChunkIterator(
        data=iter_calc_h5(os.path.join(full_path, 'FullRes', 'video-array.h5'), numZ),
        maxshape=None,
        buffer_size=10,
    )

    wrapped_data = H5DataIO(data=data, compression="gzip", compression_opts=4)

    calc_imaging_volume = ImagingVolume(
        name='CalciumImVol',
        description='Imaging volume used to acquire calcium imaging data',
        optical_channel_plus=OptChannels,
        order_optical_channels=OpticalChannelRefs,
        device=device,
        location=metadata['location'],
        grid_spacing=metadata['grid spacing'],
        grid_spacing_unit=metadata['grid spacing unit'],
        reference_frame=f"Worm {metadata['location']}"
    )

    calcium_image_series = MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        description="GCaMP6s series images. Dimensions should be (t, x, y, z, C).",
        comments="",
        data=data,
        device=device,
        unit=metadata['grid spacing unit'],
        scan_line_rate=scan_rate,
        dimension=[numX, numY, numZ],
        resolution=1.,
        rate=ai_sampling_rate,  # sampling rate in hz
        imaging_volume=calc_imaging_volume,
    )

    gcamp_mod = nwbfile.create_processing_module(
        name='GCaMPVideo',
        description='GCaMP Video'
    )
    gcamp_mod.add(calcium_image_series)
    nwbfile.add_imaging_plane(calc_imaging_volume)

    return calc_imaging_volume


def build_devices(nwbfile, metadata, count):
    for i in tqdm(range(1, int(count) + 1), desc="Building devices"):
        device_info = metadata.get(f'DEVICE_{i}', {})
        device_name = device_info.get('name', f'Unknown Device {i}')
        device_description = device_info.get('description', 'No description available')
        device_manufacturer = device_info.get('manufacturer', 'Unknown manufacturer')

        if i == 1:
            main_device = nwbfile.create_device(
                name=device_name,
                description=device_description,
                manufacturer=device_manufacturer
            )
        else:
            nwbfile.create_device(
                name=device_name,
                description=device_description,
                manufacturer=device_manufacturer
            )

    return main_device


def build_file(file_data):

    # Create NWBFile object
    nwbfile = NWBFile(
        session_description=file_data['description'],
        identifier=file_data['date'],
        session_start_time=file_data['start_time'],
        lab=file_data['lab'],
        institution=file_data['institution'],
        related_publications=file_data['related_publications']
    )

    # Create CElegansSubject object and add to NWBFile
    nwbfile.subject = CElegansSubject(
        subject_id=f"{file_data['subject']}-{file_data['strain']}",
        date_of_birth=file_data['date'],
        growth_stage=file_data['age'],
        growth_stage_time=file_data['growth_stage_time'],
        cultivation_temp=float(file_data['cultivation_temp']),
        description=file_data['notes'],
        species="C. Elegans",
        sex=file_data['sex'],
        strain=file_data['strain']
    )

    # Extract device info & build device objects
    main_device = build_devices(nwbfile, file_data.get('devices', {}), file_data.get('devices', {}).get('count', 0))

    return nwbfile, main_device


def extract_pixel_sizes(input_str):
    pattern = r"X:\s*(\d+\.\d+).*?Y:\s*(\d+\.\d+).*?Z:\s*(\d+\.\d+)"
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        x, y, z = map(float, match.groups())
        return [x, y, z]
    else:
        return None


def build_channels(metadata):
    pattern = re.compile(r'\b[A-Za-z0-9]+FP\b')
    OpticalChannels = []
    OptChanRefData = []

    # Find all matches and convert to a set to remove duplicates
    fluorophores = set(re.findall(pattern, metadata['fluorophore']))
    fluorophores = sorted(list(fluorophores))

    # Supplement with known fluorophores if they are in the text
    for fluo in ['GCaMP', 'mNeptune']:
        if fluo in metadata['fluorophore']:
            fluorophores.append(fluo)

    filters = [metadata['green cam filter'], metadata['red cam filter']]

    for eachFluo in fluorophores:
        OptChan = OpticalChannelPlus(
            name=eachFluo,
            description=eachFluo,
            excitation_lambda=0,
            excitation_range=0,
            emission_range=0,
            emission_lambda=0
        )
        OpticalChannels.append(OptChan)
        OptChanRefData.append(wave)

    """
    # Define channels and create OpticalChannelPlus objects
    channels = [("mTagBFP2", "", "405-488-50m"),
                ("CyOFP1", "", "488-594-30m"),
                ("GCaMP6s", "", "488-594-30m"),
                ("TagRFP-T", "", "594-637-70m"),
                ("mNeptune 2.5", "", "561-594-75m")]
    gcamp_channels = [("TagRFP-T", "", "594-637-70m"),
                      ("GCaMP6s", "", "488-594-30m")]

    # Create OpticalChannelReferences object
    OpticalChannelRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=OptChanRefData
    )

    gcamp_OpticalChannels = []
    gcamp_OptChanRefData = []

    for fluor, des, wave in gcamp_channels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        gcamp_OptChan = OpticalChannelPlus(
            name=fluor,
            description=des,
            excitation_lambda=excite,
            excitation_range=[excite - 1.5, excite + 1.5],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid
        )
        gcamp_OpticalChannels.append(gcamp_OptChan)
        gcamp_OptChanRefData.append(wave)

    # Create OpticalChannelReferences object
    gcamp_OpticalChannelRefs = OpticalChannelReferences(
        name='gcamp_OpticalChannelRefs',
        channels=gcamp_OptChanRefData
    )
    """
    return OpticalChannels, OpticalChannelRefs, gcamp_OpticalChannels, gcamp_OpticalChannelRefs


def build_colormap(full_path, ImagingVol, metadata, OpticalChannelRefs):
    # Load the colormap.tif file into a numpy array
    raw_file = f"{full_path}/colormap.tif"  # Replace with the actual path to your colormap.tif
    data = skio.imread(raw_file)

    # Transpose the data to the order X, Y, Z, C
    data = np.transpose(data, (3, 2, 1, 0))  # Modify as needed based on your actual data shape

    # Define RGBW channels
    RGBW_channels = [0, 1, 3, 4]

    # Create MultiChannelVolume object
    Image = MultiChannelVolume(
        name='NeuroPALImageRaw',
        order_optical_channels=OpticalChannelRefs,  # Assuming this is defined earlier
        description=f"Colormap of {metadata['identifier']}.",
        RGBW_channels=RGBW_channels,
        data=H5DataIO(data=data, compression=True),
        imaging_volume=ImagingVol  # Assuming this is defined earlier
    )
    return Image


def build_neuron_centers(full_path, ImagingVol, calc_imaging_volume):
    # Read the All_Signals.csv file
    all_signals_df = pd.read_csv(f"{full_path}/All_Signals.csv")

    # Check if Cell_ID.csv exists
    cell_id_file_path = f"{full_path}/Cell_ID.csv"
    cell_id_exists = os.path.exists(cell_id_file_path)

    if cell_id_exists:
        cell_id_df = pd.read_csv(cell_id_file_path)
        cell_id_df.columns = cell_id_df.columns.str.lower()

    colormap_center_plane = PlaneSegmentation(
        name='Neuron_Centers_colormap',
        description='Neuron segmentation associated with the colormap.',
        imaging_plane=ImagingVol,
    )

    if cell_id_exists:
        colormap_labels = []
        for i, row in tqdm(cell_id_df.iterrows(), total=cell_id_df.shape[0], desc='Processing colormap centers'):
            track_id = row['track id']
            x, y, z = row['x'], row['y'], row['z']
            try:
                neuron_name = cell_id_df[cell_id_df['track id'] == track_id]['name'].fillna('')
            except:
                neuron_name = ''
            colormap_labels.append(neuron_name)
            voxel_mask = [(np.uint(x), np.uint(y), np.uint(z), 1)]
            colormap_center_plane.add_roi(voxel_mask=voxel_mask)

        colormap_center_plane.add_column(
            name='colormap_ID_labels',
            description='ID labels corresponding to neuron centers in colormap.',
            data=colormap_labels,
            index=True
        )
        colormap_center_table = colormap_center_plane.create_roi_table_region(
            name='ROITableCentersColormap',
            description='ROIs corresponding to neuron centers in colormap.',
            region=list(np.arange(cell_id_df.shape[0])),
        )
    else:
        colormap_center_table = None

    if cell_id_exists:
        video_center_plane = PlaneSegmentation(
            name='Neuron_Centers_video',
            description='Neuron segmentation associated with the GCaMP video.',
            imaging_plane=calc_imaging_volume,
        )
    else:
        video_center_plane = PlaneSegmentation(
            name='Neuron_Centers_video',
            description='Neuron segmentation associated with the GCaMP video. No Cell ID for this run.',
            imaging_plane=calc_imaging_volume,
        )

    video_labels = []
    for i, row in tqdm(all_signals_df.iterrows(), total=all_signals_df.shape[0], desc='Processing video centers'):
        track_id = row['Track ID']
        x, y, z = row['X'], row['Y'], row['Z']
        neuron_name = ''
        if cell_id_exists:
            try:
                neuron_name = cell_id_df[cell_id_df['track id'] == track_id]['name'].fillna('')
            except:
                neuron_name = ''
        video_labels.append(neuron_name)
        voxel_mask = [(np.uint(x), np.uint(y), np.uint(z), 1)]
        video_center_plane.add_roi(voxel_mask=voxel_mask)

    if cell_id_exists:
        video_center_plane.add_column(
            name='video_ID_labels',
            description='ID labels corresponding to neuron centers in video.',
            data=video_labels,
            index=True
        )
    video_center_table = video_center_plane.create_roi_table_region(
        name='ROITableCentersVideo',
        description='ROIs corresponding to neuron centers in video.',
        region=list(np.arange(all_signals_df.shape[0])),
    )

    NeuroPALImSeg = ImageSegmentation(name='NeuroPALSegmentation')
    NeuroPALImSeg.add_plane_segmentation(colormap_center_plane)
    NeuroPALImSeg.add_plane_segmentation(video_center_plane)

    return video_center_plane, video_center_table, colormap_center_plane, colormap_center_table, NeuroPALImSeg


def build_activity(full_path, metadata, table):
    all_signals_df = pd.read_csv(f"{full_path}/All_Signals.csv")
    gcamp_data = all_signals_df['GCAMP'].to_numpy()
    timestamps = np.linspace(0, len(gcamp_data) / round(metadata['daq']['scanRate']), len(gcamp_data))
    roi_resp_series = RoiResponseSeries(
        name='GCAMP_Response',
        data=gcamp_data,
        rois=table,
        unit='lumens',
        timestamps=timestamps
    )

    # Create DfOverF
    dff_data = (gcamp_data - np.mean(gcamp_data)) / np.mean(gcamp_data)
    dff_series = DfOverF(
        name='GCAMP_readings',
        roi_response_series=roi_resp_series,
    )


def build_behavior(full_path, metadata):
    try:
        mat_file = next(f for f in os.listdir(full_path) if f.endswith('behavior.mat'))
    except StopIteration:
        print("-NO BEHAVIOR FOUND.")
        return None

    mat_data = scipy.io.loadmat(os.path.join(full_path, mat_file))

    try:
        behavioral_data = mat_data['behavior_all']
    except:
        behavioral_data = mat_data['behavior_total']

    # Initialize an empty set to store unique frame numbers
    frame_set = set()

    # List all files in the folder
    files = os.listdir(os.path.join(full_path, 'FullRes'))

    # Regular expression to match the filename pattern
    regex_pattern = r"worm\d+_run\d+_t(\d+)_\w\.tiff"

    for file in files:
        match = re.match(regex_pattern, file)
        if match:
            frame_number = int(match.group(1))
            frame_set.add(frame_number)

    behavior = SpatialSeries(
        name='Coded_behavior',
        description=metadata['behavior']['description'],
        comments='',
        data=behavioral_data,
        reference_frame='See description.',
        timestamps=sorted(list(frame_set)),
    )

    return behavior


def build_nwb(nwbfile, datapath, metadata, main_device):
    #behavior = build_behavior(full_path, metadata)
    OpticalChannels, OpticalChannelRefs, gcamp_OpticalChannels, gcamp_OpticalChannelRefs = build_channels(metadata)
    location = metadata['location']

    # Create ImagingVolume object
    ImagingVol = ImagingVolume(
        name='NeuroPALImVol',
        optical_channel_plus=OpticalChannels,
        order_optical_channels=OpticalChannelRefs,
        description='NeuroPAL image of C. elegans',
        device=main_device,
        location=location,
        grid_spacing=extract_pixel_sizes(metadata['comments']),
        grid_spacing_unit='micrometers',
        origin_coords=[0, 0, 0],
        origin_coords_unit='micrometers',
        reference_frame=f"Worm {location}"
    )

    Image = build_colormap(datapath, ImagingVol, metadata, OpticalChannelRefs)
    nwbfile.add_imaging_plane(ImagingVol)

    processed_module = nwbfile.create_processing_module(
        name='Processed',
        description='Processed image data and metadata.'
    )

    # Discover and sort tiff files, build single .h5 file for iterator compatibility.
    calc_imaging_volume = build_gcamp(nwbfile, datapath, gcamp_OpticalChannels, gcamp_OpticalChannelRefs, main_device,
                                      metadata)

    video_center_plane, video_center_table, colormap_center_plane, colormap_center_table, NeuroPALImSeg = build_neuron_centers(
        datapath, ImagingVol, calc_imaging_volume)
    build_activity(datapath, metadata, video_center_table)
    #build_mip(nwbfile, ImagingVol, full_path)

    processed_module.add(Image)
    processed_module.add(NeuroPALImSeg)
    processed_module.add(OpticalChannelRefs)
    processed_module.add(gcamp_OpticalChannelRefs)
    #if behavior is not None:
    #    processed_module.add(behavior)

    # specify the file path you want to save this NWB file to
    save_path = datapath + ".nwb"
    io = NWBHDF5IO(save_path, mode='w')
    io.write(nwbfile)
    io.close()


if __name__ == "__main__":
    iterate_folders()

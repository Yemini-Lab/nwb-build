import scipy.io as sio
import configparser
import datetime
import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict
import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import skimage.io as skio
import typing_extensions
from dateutil import tz
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, \
    MultiChannelVolume, MultiChannelVolumeSeries, SegmentationLabels
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position, BehavioralTimeSeries, BehavioralEvents
from pynwb.ophys import ImageSegmentation, PlaneSegmentation, \
    DfOverF, RoiResponseSeries, Fluorescence
from pynwb.image import ImageSeries
from tifffile import TiffFile
from tifffile import imread
from tqdm import tqdm
import nd2reader
import nrrd
from pathlib import Path
from statistics import mean


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
    shape = (shape['t'], shape['c'], 1, shape['y'], shape['x'])
    h5_file.create_dataset('dataset', shape, dtype='uint16')

    try:
        print("Populating the .h5 file...")
        for i in tqdm(range(shape[0]), desc="Processing frames"):
            green_data = nd2_file.get_frame_2D(t=i, c=1)
            red_data = nd2_file.get_frame_2D(t=i, c=2)
            print(np.shape(green_data))
            print(np.shape(h5_file['dataset'][i, 1, 1, :, :]))
            h5_file['dataset'][i, 1, 1, :, :] = green_data
            h5_file['dataset'][i, 2, 1, :, :] = red_data
    finally:
        print("Flushing changes and closing the file.")
        h5_file.close()


def iter_calc_h5(file_path):
    nd2_file = nd2reader.ND2Reader(file_path)
    f_shape = (nd2_file.sizes['t'], nd2_file.sizes['c'], 1, nd2_file.sizes['y'], nd2_file.sizes['x'])
    for i in tqdm(range(int(f_shape[0] / 80)), desc="Processing time points"):
        tpoint = np.zeros((nd2_file.sizes['y'], nd2_file.sizes['x'], 80, nd2_file.sizes['c']), dtype='uint16')
        for j in range(80):
            tpoint[:, :, j, 0] = nd2_file.get_frame_2D(t=i * 80 + j, c=0)
            tpoint[:, :, j, 1] = nd2_file.get_frame_2D(t=i * 80 + j, c=1)

        yield np.squeeze(np.transpose(tpoint, [1, 0, 2, 3]))

    """
    with h5py.File(filename, 'r') as h5_file:
        t, x, y, z, c = h5_file['dataset'].shape

        for i in tqdm(range(t), desc="Processing time points"):
            tpoint = np.zeros((x, y, z, c), dtype='uint16')
            for j in range(numZ):
                tpoint[:, :, j, :] = h5_file['dataset'][i, :, :, j, :]

            yield np.squeeze(tpoint)
    """


# Build bright field NIR
def build_nir(nwbfile, video_path, behavior_timestamps):
    with h5py.File(video_path, 'r') as hdf:
        data = hdf['img_nir'][:]
        timestamps = hdf['img_metadata']['img_timestamp'][:]

    decimal_location = str(behavior_timestamps[0]).index('.')

    timestamps = np.asarray(
        [float(str(time)[0:decimal_location] + '.' + str(time)[decimal_location:]) for time in timestamps])

    nir_data = np.array(data)
    nir_data = np.transpose(data, axes=(0, 2, 1))  # Shape should be (T, X, Y)
    nir_data = nir_data[:, :, :]
    hefty_data = H5DataIO(data=nir_data, compression=True)

    nir_vol_series = ImageSeries(
        name='BrightFieldNIR',
        description='The light path used to image behavior was in a reflected brightfield (NIR) configuration. Light '
                    'supplied by an 850-nm LED (M850L3, Thorlabs) was collimated and passed through an 850/10 '
                    'bandpass filter (FBH850-10, Thorlabs). Illumination light was reflected towards the sample by a '
                    'half mirror and was focused on the sample through a 10x objective (CFI Plan Fluor 10x, '
                    'Nikon). The image from the sample passed through the half mirror and was filtered by another '
                    '850-nm bandpass filter of the same model. The image was captured by a CMOS camera ('
                    'BFS-U3-28S5M-C, FLIR).',
        data=hefty_data,
        unit='',
        timestamps=timestamps[::2],
        # timestamps=list(range(hefty_data.shape[0])),
    )

    nir_module = nwbfile.create_processing_module(
        name='BF_NIR',
        description='The light path used to image behavior was in a reflected brightfield (NIR) configuration. Light '
                    'supplied by an 850-nm LED (M850L3, Thorlabs) was collimated and passed through an 850/10 '
                    'bandpass filter (FBH850-10, Thorlabs). Illumination light was reflected towards the sample by a '
                    'half mirror and was focused on the sample through a 10x objective (CFI Plan Fluor 10x, '
                    'Nikon). The image from the sample passed through the half mirror and was filtered by another '
                    '850-nm bandpass filter of the same model. The image was captured by a CMOS camera ('
                    'BFS-U3-28S5M-C, FLIR).'
    )
    nir_module.add(nir_vol_series)

    return nir_vol_series


def build_gcamp(nwbfile, package, OptChannels, OpticalChannelRefs, device):
    nd2_file, frames, channels = discover_nd2_files(os.path.join(f"{full_path}/{metadata['subject']}.nd2"))
    # h5_memory_mapper(nd2_file, os.path.join(full_path, f'{metadata["subject"]}-array.h5'))

    scan_rate = 2.66
    ai_sampling_rate = 0

    numX = nd2_file.sizes['x']
    numY = nd2_file.sizes['y']

    # Create DataChunkIterator
    data = DataChunkIterator(
        data=iter_calc_h5(package['data']['video']['raw_path']),
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
        data=wrapped_data,
        device=device,
        unit=metadata['grid spacing unit'],
        scan_line_rate=scan_rate,
        dimension=[numX, numY],
        resolution=1.,
        rate=ai_sampling_rate,  # sampling rate in hz
        imaging_volume=calc_imaging_volume,
    )

    nwbfile.add_acquisition(calcium_image_series)
    nwbfile.add_imaging_plane(calc_imaging_volume)

    return calc_imaging_volume


def build_devices(nwbfile, package):
    main_device = nwbfile.create_device(
        name='LAMBDA',
        description='https://github.com/venkatachalamlab/lambda',
        manufacturer='Venkatachalam Laboratory'
    )

    return main_device


def build_file(package):
    print("- Building NWBFile object...")

    nwbfile = NWBFile(
        session_description=package['metadata']['description'],
        identifier=package['metadata']['identifier'],
        session_start_time=package['metadata']['date'],
        lab=package['metadata']['lab'],
        institution=package['metadata']['institution'],
        related_publications=package['metadata']['related_publications']
    )
    print("- NWBFile object created.")

    print("- Creating CElegansSubject...")

    nwbfile.subject = CElegansSubject(
        subject_id=package['metadata']['identifier'],
        date_of_birth=package['metadata']['date'] - timedelta(days=3),
        growth_stage=package['metadata']['age'],
        growth_stage_time=str(np.nan),
        cultivation_temp=float(package['metadata']['cultivation_temp'][:-1]),
        description=str(package['metadata']['notes']),
        species="http://purl.obolibrary.org/obo/NCBITaxon_6239",
        sex=package['metadata']['sex'],
        strain=str(package['metadata']['strain'])
    )
    print("- CElegansSubject object added to NWBFile.")

    print("Extracting device info and building device objects...")
    main_device = build_devices(nwbfile, package)
    print("Device objects built and added to NWBFile.")

    return nwbfile, main_device


def extract_pixel_sizes(input_str):
    pattern = r"X:\s*(\d+\.\d+).*?Y:\s*(\d+\.\d+).*?Z:\s*(\d+\.\d+)"
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        x, y, z = map(float, match.groups())
        return [x, y, z]
    else:
        return None


def build_channels(package):
    print("-- Starting to build channels...")
    OpticalChannels = []
    OptChanRefData = []

    fluo_dict = {
        'mNeptune': {
            'ex_lambda': 600,
            'ex_range': [451, 658],
            'em_lambda': 650,
            'em_range': [542, 763],
            'link': 'https://www.fpbase.org/protein/mneptune/'
        },
        'BFP': {
            'ex_lambda': 381,
            'ex_range': [0, 0],
            'em_lambda': 445,
            'em_range': [0, 0],
            'link': 'https://www.fpbase.org/protein/bfp/'
        },
        'CyOFP': {
            'ex_lambda': 497,
            'ex_range': [349, 591],
            'em_lambda': 589,
            'em_range': [525, 715],
            'link': 'https://www.fpbase.org/protein/cyofp1/'
        },
        'RFP': {
            'ex_lambda': 558,
            'ex_range': [365, 581],
            'em_lambda': 583,
            'em_range': [523, 750],
            'link': 'https://www.fpbase.org/protein/dsred/'
        },
        'GCaMP': {
            'ex_lambda': 487,
            'ex_range': [0, 0],
            'em_lambda': 508,
            'em_range': [0, 0],
            'link': 'https://www.fpbase.org/protein/gcamp2/'
        },
    }

    # Channels is a list of tuples where each tuple contains the fluorophore used, the specific emission filter used, and a short description
    # structured as "excitation wavelength - emission filter center point- width of emission filter in nm"
    # Make sure this list is in the same order as the channels in your data
    for eachFluo in fluo_dict.keys():
        OptChan = OpticalChannelPlus(
            name=eachFluo,
            description=fluo_dict[eachFluo]['link'],
            excitation_lambda=float(fluo_dict[eachFluo]['ex_lambda']),
            excitation_range=fluo_dict[eachFluo]['ex_range'],
            emission_range=fluo_dict[eachFluo]['em_range'],
            emission_lambda=float(fluo_dict[eachFluo]['em_lambda'])
        )
        OpticalChannels.append(OptChan)
        OptChanRefData.append(
            f"{fluo_dict[eachFluo]['ex_lambda']}-{mean(fluo_dict[eachFluo]['em_range'])}-{fluo_dict[eachFluo]['em_range'][1] - fluo_dict[eachFluo]['em_range'][0]}nm")

    print("-- Channels built.")

    # Create OpticalChannelReferences object
    OpticalChannelRefs = OpticalChannelReferences(
        name='order_optical_channels',
        channels=OptChanRefData
    )

    print("-- OpticalChannelReferences created.")
    return OpticalChannels, OpticalChannelRefs


# Convert each string into a tuple of integers, handling non-numeric characters
def parse_coordinate(coord):
    # Replace non-numeric characters (except commas) with an empty string
    cleaned = ''.join(c if c.isdigit() or c == ',' or c == '.' else '' for c in coord)
    if len(cleaned) < 5:
        return None
    return tuple(map(int, cleaned.split(',')))


def build_colormap(package, ImagingVol, order_optical_channels):
    print("-- Building NeuroPAL volume...")

    # Transpose the data to the order X, Y, Z, C
    dims = np.shape(package['data']['map']['contents'])
    if dims[3] != min(dims):
        package['data']['map']['contents'] = np.transpose(package['data']['map']['contents'], (3, 2, 1, 0))

    # Create MultiChannelVolume object
    Image = MultiChannelVolume(
        name='NeuroPALImageRaw',
        order_optical_channels=order_optical_channels,
        description=f"NeuroPAL volume of {package['metadata']['animal_id']}.",
        RGBW_channels=package['metadata']['map']['RGBW'],
        data=H5DataIO(data=package['data']['map']['contents'], compression=True),
        imaging_volume=ImagingVol
    )

    print("-- Successfully built NeuroPAL volume.")
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
            index=False
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
            index=False
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


def build_activity(data_path, file_name, calc_imaging_volume, labels, metadata, timestamps):
    nd2_file, frames, channels = discover_nd2_files(os.path.join(f"{data_path}/{metadata['subject']}.nd2"))
    # h5_memory_mapper(nd2_file, os.path.join(full_path, f'{metadata["subject"]}-array.h5'))

    scan_rate = 1.7 / nd2_file.sizes['y']
    scan_rate = 1 / scan_rate
    ai_sampling_rate = 1.7

    with open(f"{data_path}/../processed/{file_name}.json", 'r') as file:
        json_data = json.load(file)
        gcamp_data = json_data.get('trace_original')
        neurons = json_data.get('labeled')

        calc_labels = [''] * len(gcamp_data[0])

        calc_coords = PlaneSegmentation(
            name='Aligned_neuron_coordinates',
            description='Neuron center coordinates in aligned space',
            imaging_plane=calc_imaging_volume
        )

        CalcImSeg = ImageSegmentation(
            name='CalciumSeriesSegmentation',
        )

        CalcImSeg.add_plane_segmentation(calc_coords)

        for i in range(len(calc_labels)):
            if str(i + 1) in neurons.keys():
                calc_labels[i] = neurons[str(i + 1)]['label']
                coordinates = parse_coordinate(
                    np.asarray(labels[labels['Class'] == neurons[str(i + 1)]['label']]['Coordinates '])[0])
                calc_coords.add_roi(voxel_mask=[[coordinates[0], coordinates[1], coordinates[2] - 1, 1]])

            else:
                calc_coords.add_roi(voxel_mask=[[0, 0, 0, 0]])

        calc_coords.add_column(
            name='ID_labels',
            description='Neuron Names',
            data=calc_labels,
            index=False,
        )

        CalcLabels = SegmentationLabels(
            name='NeuronIDs',
            labels=calc_labels,
            description='Calcium ROI segmentation labels',
            ImageSegmentation=CalcImSeg
        )

        rt_region = calc_coords.create_roi_table_region(
            description='Segmented neurons associated with calcium image series',
            region=list(np.arange(calc_coords.voxel_mask.shape[0]))
        )

        # timestamps = np.linspace(0, len(gcamp_data) / ai_sampling_rate, len(gcamp_data))
        SignalRoiResponse = RoiResponseSeries(
            name='SignalCalciumImResponseSeries',
            description='Raw calcium fluorescence activity',
            data=gcamp_data,
            rois=rt_region,
            unit='lumens',
            timestamps=timestamps
        )

        SignalFluor = Fluorescence(
            name='SignalRawFluor',
            roi_response_series=SignalRoiResponse,
        )

    return SignalRoiResponse, SignalFluor, CalcLabels, calc_coords, CalcImSeg


def build_nwb(nwb_file, package, main_device):
    print("Starting to build NWB for run...")

    OpticalChannels, order_optical_channels = build_channels(package)

    neuroPAL_module = nwb_file.create_processing_module(
        name='NeuroPAL',
        description='NeuroPAL image metadata and segmentation'
    )

    print("- Creating ImagingVolume object for NeuroPAL...")  # Debug print
    ImagingVol = ImagingVolume(
        name='NeuroPALImVol',
        optical_channel_plus=OpticalChannels,
        order_optical_channels=order_optical_channels,
        description='NeuroPAL image of C. Elegans',
        device=main_device,
        location=package['metadata']['location'],
        grid_spacing=package['metadata']['map']['grid_spacing'],
        grid_spacing_unit='micrometer',
        origin_coords=[0, 0, 0],
        origin_coords_unit='micrometer',
        reference_frame=f"Worm {package['metadata']['location']}"
    )

    Image = build_colormap(package, ImagingVol, order_optical_channels)

    nwb_file.add_imaging_plane(ImagingVol)
    nwb_file.add_acquisition(Image)

    print("Processing neuron ROIs...")


    vs = PlaneSegmentation(
        name='NeuronSegmentationROIs',
        description='Segmentation of NeuroPAL volume. IDs found in NeuroPALNeurons.',
        imaging_plane=ImagingVol,
    )

    neurons = pd.read_csv(Path(package['data']['map']['volume']).parent / 'data.csv', header=5)

    roi_ids = []

    for idx, row in neurons.iterrows():
        vs.add_roi(voxel_mask=[[row['Real X (um)'], row['Real Y (um)'], row['Real Z (um)'], 1]])
        roi_ids.append(row['User ID'])

    vs.add_column(
        name='Neuron_IDs',
        description='Neuron ID labels from segmentation image mask.',
        data=roi_ids,
        index=False,
    )

    NeuroPALImSeg = ImageSegmentation(
        name='NeuroPALSegmentation',
    )

    NeuroPALImSeg.add_plane_segmentation(vs)
    neuroPAL_module.add(NeuroPALImSeg)

    print("Neuron ROIs processed and added to NWBFile.")

    ophys = nwb_file.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )

    # Discover and sort tiff files, build single .h5 file for iterator compatibility.
    calc_imaging_volume = build_gcamp(nwb_file, package, OpticalChannels, order_optical_channels, main_device)

    # video_center_plane, video_center_table, colormap_center_plane, colormap_center_table, NeuroPALImSeg = build_neuron_centers(
    #    data_path, ImagingVol, calc_imaging_volume)
    signal_roi, signal_fluor, calc_labels, calc_volseg, calc_imseg = build_activity(data_path, file_name,
                                                                                    calc_imaging_volume, labels,
                                                                                    metadata, timestamps)

    ophys.add(calc_imseg)
    ophys.add(signal_roi)
    ophys.add(signal_fluor)
    ophys.add(calc_labels)

    for eachBehavior in behavior:
        behavior_module.add(eachBehavior)

    # specify the file path you want to save this NWB file to
    save_path = f"{data_path}/../../NWB_flavell_final_2024_03_19/{file_name}.nwb"
    # save_path = f"{data_path}/../../NWB_NP_flavell/{file_name}.nwb"
    # save_path = f"/Users/danielysprague/foco_lab/data/NWB_test/{file_name}.nwb"
    io = NWBHDF5IO(save_path, mode='w')
    io.write(nwb_file)
    io.close()
    print("NWB file built and saved at:", save_path)  # Debug print


if __name__ == "__main__":
    iterate_folders()

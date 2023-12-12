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
import typing_extensions
from dateutil import tz
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume, \
    MultiChannelVolume, MultiChannelVolumeSeries
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position, BehavioralTimeSeries, BehavioralEvents
from pynwb.ophys import ImageSegmentation, PlaneSegmentation, \
    DfOverF, RoiResponseSeries
from tifffile import TiffFile
from tifffile import imread
from tqdm import tqdm
import nd2reader
import nrrd
from pathlib import Path


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
    print(shape)
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
    for i in tqdm(range(f_shape[0]), desc="Processing time points"):
        tpoint = np.zeros((nd2_file.sizes['y'], nd2_file.sizes['x'], 1, nd2_file.sizes['c']), dtype='uint16')
        tpoint[:, :, 0, 0] = nd2_file.get_frame_2D(t=i, c=0)
        tpoint[:, :, 0, 1] = nd2_file.get_frame_2D(t=i, c=1)

        yield np.squeeze(tpoint)

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
def build_nir(nwbfile, ImagingVol, video_path):
    with h5py.File(video_path, 'r') as hdf:
        data = hdf['img_nir'][:]

    nir_data = np.array(data)
    nir_data = np.transpose(data, axes=(0, 2, 1))  # Shape should be (T, X, Y)
    nir_data = nir_data[:, np.newaxis, :, :, np.newaxis]  # Reshape to (T, X, Y, 1, 1) to fit (T, X, Y, Z, C)
    hefty_data = H5DataIO(data=nir_data, compression=True)

    nir_vol_series = MultiChannelVolumeSeries(
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
        timestamps=list(range(hefty_data.shape[0])),
        imaging_volume=ImagingVol
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


def build_gcamp(nwbfile, full_path, OptChannels, OpticalChannelRefs, device, metadata):
    nd2_file, frames, channels = discover_nd2_files(os.path.join(f"{full_path}\\{metadata['subject']}.nd2"))
    #h5_memory_mapper(nd2_file, os.path.join(full_path, f'{metadata["subject"]}-array.h5'))

    print(nd2_file.sizes)
    scan_rate = 1.7/nd2_file.sizes['y']
    ai_sampling_rate = 1.7

    numX = nd2_file.sizes['x']
    numY = nd2_file.sizes['y']

    # Create DataChunkIterator
    data = DataChunkIterator(
        data=iter_calc_h5(os.path.join(f"{full_path}\\{metadata['subject']}.nd2")),
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
        dimension=[numX, numY],
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


def build_devices(nwbfile, metadata):
    main_device = nwbfile.create_device(
            name=metadata['devices']['vol']['name'],
            description=metadata['devices']['vol']['description'],
            manufacturer=metadata['devices']['vol']['manufacturer']
        )
    nir_device = nwbfile.create_device(
            name=metadata['devices']['nir']['name'],
            description=metadata['devices']['nir']['description'],
            manufacturer=metadata['devices']['nir']['manufacturer']
        )

    return main_device, nir_device


def build_file(file_info):
    print("Starting to build NWBFile object...")  # Debug print
    file_data = file_info['metadata']

    # Create NWBFile object
    nwbfile = NWBFile(
        session_description=file_data[0]['notes'],
        identifier=str(file_data[0]['date']),
        session_start_time=file_data[0]['date'],
        lab=file_info['lab'],
        institution=file_info['institution'],
        related_publications=file_info['related_publications']
    )
    print("NWBFile object created.")  # Debug print

    print("Creating CElegansSubject object...")  # Debug print
    # Create CElegansSubject object and add to NWBFile
    nwbfile.subject = CElegansSubject(
        subject_id=f"{file_data[0]['subject']}-{file_data[0]['strain']}",
        date_of_birth=file_data[0]['date'],
        growth_stage=file_data[0]['age'],
        growth_stage_time=str(np.nan),
        cultivation_temp=float(file_info['cultivation_temp'][:-1]),
        description=file_data[0]['notes'],
        species="C. Elegans",
        sex=file_data[0]['sex'],
        strain=file_data[0]['strain']
    )
    print("CElegansSubject object added to NWBFile.")  # Debug print

    print("Extracting device info and building device objects...")  # Debug print
    # Extract device info & build device objects
    main_device, nir_device = build_devices(nwbfile, file_info)
    print("Device objects built and added to NWBFile.")  # Debug print

    return nwbfile, main_device, nir_device


def extract_pixel_sizes(input_str):
    pattern = r"X:\s*(\d+\.\d+).*?Y:\s*(\d+\.\d+).*?Z:\s*(\d+\.\d+)"
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        x, y, z = map(float, match.groups())
        return [x, y, z]
    else:
        return None


def build_channels(metadata):
    print("Starting to build channels...")
    pattern = re.compile(r'\b[A-Za-z0-9]+FP\b')
    OpticalChannels = []
    OptChanRefData = []

    excitation_dict = {
        'OFP': {'ex_lambda': float(488), 'em_lambda': float(589), 'cam': 'red'},
        'TagRFP': {'ex_lambda': float(561), 'em_lambda': float(584), 'cam': 'red'},
        'BFP': {'ex_lambda': float(405), 'em_lambda': float(445), 'cam': 'red'},
        'mNeptune': {'ex_lambda': float(637), 'em_lambda': float(650), 'cam': 'red'},
        'GCaMP': {'ex_lambda': float(488), 'em_lambda': float(513), 'cam': 'green'},
    }

    # Find all matches and convert to a set to remove duplicates
    fluorophores = set(re.findall(pattern, metadata['fluorophore']))
    fluorophores = sorted(list(fluorophores))

    # Supplement with known fluorophores if they are in the text
    for fluo in ['GCaMP', 'mNeptune']:
        if fluo in metadata['fluorophore']:
            fluorophores.append(fluo)

    print(f"Identified fluorophores: {fluorophores}")

    filters = {}

    for eachCam in ['green', 'red']:
        for note in [' ', '+ NDF', '-TRF']:
            waves = metadata[f'{eachCam} cam filter'].replace(note, '')

        if '/' in metadata[f'{eachCam} cam filter']:
            waves = waves.split('/')

            filters[eachCam] = {
                'c_point': float(waves[0]),
                'width': float(waves[1]),
            }
        else:
            if 'LP' in metadata[f'{eachCam} cam filter']:
                waves = waves[:-2]

                filters[eachCam] = {
                    'c_point': float(waves),
                    'width': np.nan,
                }

    print(f"Camera filters processed: {filters}")

    # Channels is a list of tuples where each tuple contains the fluorophore used, the specific emission filter used, and a short description
    # structured as "excitation wavelength - emission filter center point- width of emission filter in nm"
    # Make sure this list is in the same order as the channels in your data
    for eachFluo in fluorophores:
        fluo_cam = filters[excitation_dict[eachFluo]['cam']]
        if fluo_cam['width'] != np.nan:
            OptChan = OpticalChannelPlus(
                name=eachFluo,
                description=eachFluo,
                excitation_lambda=excitation_dict[eachFluo]['ex_lambda'],
                excitation_range=[excitation_dict[eachFluo]['ex_lambda'] - 1.5,
                                  excitation_dict[eachFluo]['ex_lambda'] + 1.5],
                emission_range=[fluo_cam['c_point'] - fluo_cam['width'] / 2,
                                fluo_cam['c_point'] + fluo_cam['width'] / 2],
                emission_lambda=excitation_dict[eachFluo]['em_lambda']
            )
            OpticalChannels.append(OptChan)
            OptChanRefData.append(
                f"{excitation_dict[eachFluo]['ex_lambda']}-{fluo_cam['c_point']}-{fluo_cam['width']}nm")
        else:
            OptChan = OpticalChannelPlus(
                name=eachFluo,
                description=eachFluo,
                excitation_lambda=excitation_dict[eachFluo]['ex_lambda'],
                excitation_range=[excitation_dict[eachFluo]['ex_lambda'] - 1.5,
                                  excitation_dict[eachFluo]['ex_lambda'] + 1.5],
                emission_range=[fluo_cam['c_point'] - fluo_cam['width'] / 2, np.nan],
                emission_lambda=excitation_dict[eachFluo]['em_lambda']
            )
            OpticalChannels.append(OptChan)
            OptChanRefData.append(f"{excitation_dict[eachFluo]['ex_lambda']}-{fluo_cam['c_point']}LP")


    print("Channels built.")

    # Create OpticalChannelReferences object
    OpticalChannelRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=OptChanRefData
    )

    print("OpticalChannelReferences created.")
    return OpticalChannels, OpticalChannelRefs

# Convert each string into a tuple of integers, handling non-numeric characters
def parse_coordinate(coord):
    # Replace non-numeric characters (except commas) with an empty string
    cleaned = ''.join(c if c.isdigit() or c == ',' else '' for c in coord)
    return tuple(map(int, cleaned.split(',')))


def build_colormap(full_path, file_name, ImagingVol, metadata, OpticalChannelRefs):
    print("Building NeuroPAL volume...")

    file, header = nrrd.read(f"{full_path}/prj_neuropal/{file_name}/NeuroPAL.nrrd", index_order='C')

    # Transpose the data to the order X, Y, Z, C
    data = np.transpose(file, (3, 2, 1, 0))  # Modify as needed based on your actual data shape

    # Create MultiChannelVolume object
    Image = MultiChannelVolume(
        name='NeuroPALImageRaw',
        order_optical_channels=OpticalChannelRefs,  # Assuming this is defined earlier
        description=f"NeuroPAL volume of {metadata['metadata'][0]['subject']}.",
        RGBW_channels=list(range(min(header['sizes']))),
        data=H5DataIO(data=data, compression=True),
        imaging_volume=ImagingVol
    )
    print("Successfully built NeuroPAL volume.")
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


def build_behavior(data_path, file_name, metadata):
    with open(f"{data_path}\\processed\\{file_name}.json", 'r') as file:
        json_data = json.load(file)

        behavior_dict = {
            'dorsalness': {'type':'time series','description':'The dorsalness metric is computed similarly to the forwardness metric.'},
            'head_curvature': {'type':'time series','description':'Head curvature is computed as the angle between the points 1, 5, and 8 (ie: the angle between θ→_1,5 and θ→_5,8 ). These points are 0 μm, 35.4 μm, and 61.9 μm along the worm’s spline, respectively.'},
            'angular_velocity': {'type':'time series','description':'Angular velocity is computed as smoothed (dθ→_1,2)/(dt) , which is computed with a linear Savitzky-Golay filter with a width of 300 time points (15 seconds) centered on the current time point.'},
            'reversal_events': {'type':'event','description':''},
            'feedingness': {'type':'time series','description':'The feedingness metric is computed similarly to the forwardness metric.'},
            'velocity': {'type':'time series','description':'First, we read out the (x,y) position of the stage (in mm) as it tracks the worm. To account for any delay between the worm’s motion and stage tracking, at each time point we added the distance from the center of the image (corresponding to the stage position) to the position of the metacorpus of pharynx (detected from our neural network used in tracking). This then gave us the position of the metacorpus over time. To decrease the noise level (e.g., from neural network and stage jitter), we then applied a Group-Sparse Total-Variation Denoising algorithm to the metacorpus position. Differentiating the metacorpus position then gives us a movement vector of the animal. Because this movement vector was computed from the location of the metacorpus, it contains two components of movement: the animal’s velocity in its direction of motion, and oscillations of the animal’s head perpendicular to that direction. To filter out these oscillations, we projected the movement vector onto the animal’s facing direction, i.e. the vector from the grinder of the pharynx to its metacorpus (computed from the stage-tracking neural network output). The result of this projection is a signed scalar, which is reported as the animal’s velocity.'},
            'body_curvature': {'type':'time series','description':'Body curvature is computed as the standard deviation of θ→_(i, i+1) for i between 1 and 31 (ie: going up to 265 μm along the worm’s spline). This value was selected such that this length of the animal would almost never be cropped out of the NIR camera’s field of view. To ensure that these angles are continuous in i , they may each have 2pi added or subtracted as appropriate.'},
            'forwardness': {'type':'time series','description':"The forwardness metric for a neuron class is computed as F_D * (σM/σD) * signal, where F_D is the deconvolved forwardness of the Cartesian average μ_cart of the hierarchical model fit to that neuron class (see “deconvolved activity matrix” and “hierarchical model” methods sections above for more details; the behavior values used in the deconvolved forwardness computation were constructed by appending together all of the behaviors for the neuron class), σD is the standard deviation of the model fit corresponding to μ_cart with s = 0, σM is the standard deviation of the model fit corresponding to μ_cart, σD, and signal is defined as in the “Statistical encoding tests” section of the related publication. This ratio is intended to correct for the fact that the model parameters need to be larger (resulting in larger deconvolved forwardness values) for the same neural response size if the neuron has a long EWMA decay."},
            'pumping': {'type':'time series','description':'Pumping rate was manually annotated using Datavyu, by counting each pumping stroke while watching videos slowed down the 25% of their real-time speeds. The rate is then filtered via a moving average with a width of 80 time points (4 seconds) to smoothen the trace into a pumping rate rather than individual pumping strokes.'},
        }

        behavior = []

        for eachBehavior in behavior_dict.keys():
            description = behavior_dict[eachBehavior]['description']
            data = json_data.get(eachBehavior)
            timestamps = json_data.get('timestamp_confocal')
            if behavior_dict[eachBehavior]['type'] == 'time series':
                thisBehavior = BehavioralTimeSeries(name=eachBehavior)
                thisBehavior.create_timeseries(name=eachBehavior,
                                                data=data,
                                                timestamps=timestamps,
                                                unit='')
            elif behavior_dict[eachBehavior]['type'] == 'event':
                ts = np.zeros(np.shape(timestamps))

                for eachEvent in data:
                    start = eachEvent[0]
                    end = eachEvent[1]
                    ts[start:end] = 1

                thisBehavior = BehavioralEvents(name=eachBehavior)
                thisBehavior.create_timeseries(name=eachBehavior,
                                                data=ts,
                                                timestamps=timestamps,
                                                unit='')
            elif behavior_dict[eachBehavior]['type'] == 'coded':
                thisBehavior = SpatialSeries(
                    name=eachBehavior,
                    description=behavior_dict[eachBehavior]['description'],
                    data=data
                )

            behavior += [thisBehavior]

        return behavior


def build_nwb(nwb_file, file_info, run, main_device, nir_device):
    print("Starting to build NWB for run:", run)  # Debug print

    data_path = Path(file_info['path']).parent
    file_name = f"{file_info['date']}-0{run+1}"
    metadata = file_info['metadata'][run]
    metadata['grid spacing'] = [0.54, 0.54, 0.54]
    metadata['grid spacing unit'] = 'um'

    nd2_path = f"{data_path}/{file_name}.nd2"
    h5_path = f"{data_path}/{file_name}.h5"

    OpticalChannels, OpticalChannelRefs = build_channels(metadata)
    behavior = build_behavior(data_path, file_name, metadata)

    if os.path.exists(f"{data_path}/prj_neuropal/{file_name}/NeuroPAL.nrrd"):
        print("Creating ImagingVolume object for NeuroPAL...")  # Debug print
        ImagingVol = ImagingVolume(
            name='NeuroPALImVol',
            optical_channel_plus=OpticalChannels,
            order_optical_channels=OpticalChannelRefs,
            description='NeuroPAL image of C. elegans',
            device=main_device,
            location=metadata['location'],
            grid_spacing=metadata['grid spacing'],
            grid_spacing_unit=metadata['grid spacing unit'],
            origin_coords=[0, 0, 0],
            origin_coords_unit=metadata['grid spacing unit'],
            reference_frame=f"Worm {metadata['location']}"
        )

        Image = build_colormap(data_path, file_name, ImagingVol, file_info, OpticalChannelRefs)
        nwb_file.add_imaging_plane(ImagingVol)
    else:
        print('No NeuroPAL found.')  # Debug print

    if os.path.exists(f"{data_path}/prj_neuropal/{file_name}/neuron_rois.nrrd"):
        print("Processing neuron ROIs...")  # Debug print
        file, header = nrrd.read(f"{data_path}/prj_neuropal/{file_name}/neuron_rois.nrrd", index_order='C')
        labels = pd.read_excel(f"{data_path}/prj_neuropal/{file_name}/{file_name} Neuron ID.xlsx")
        #print(np.shape(file))
        data = np.transpose(file, (2, 1, 0))

        vs = PlaneSegmentation(
            name='NeuronSegmentationROIs',
            description='Segmentation of NeuroPAL volume. IDs found in NeuronCenters.',
            imaging_plane=ImagingVol,
        )

        vs.add_roi(image_mask=data)

        coord_base = PlaneSegmentation(
            name='NeuronCenters',
            description='Centers of NeuroPAL volume.',
            imaging_plane=ImagingVol,
        )

        coordinates = [parse_coordinate(coord) for coord in labels['Coordinates ']]

        for neuron in coordinates:
            coord_base.add_roi(pixel_mask=[(neuron[0], neuron[1], neuron[2])])

        roi_ids = []
        for item in list(labels['ROI ID']):
            if isinstance(item, str):
                first_int = int(item.split()[0])
                roi_ids.append(first_int)
            else:
                roi_ids.append(item)

        coord_base.add_column(
            name='ROI_ID_labels',
            description='ROI ID labels from segmentation image mask.',
            data=roi_ids,
            index=False,
        )

        coord_base.add_column(
            name='Neuron_Names',
            description='Neuron Names',
            data=list(labels['Class']),
            index=False,
        )

        NeuroPALImSeg = ImageSegmentation(
            name='NeuroPALSegmentation',
        )

        NeuroPALImSeg.add_plane_segmentation(vs)
        NeuroPALImSeg.add_plane_segmentation(coord_base)

        # Create ImagingVolume object
        ImagingVol = ImagingVolume(
            name='NeuronSegmentation',
            optical_channel_plus=OpticalChannels,
            order_optical_channels=OpticalChannelRefs,
            description='Neuron Centers & Neuron ROIs.',
            device=main_device,
            location=metadata['location'],
            grid_spacing=metadata['grid spacing'],
            grid_spacing_unit=metadata['grid spacing unit'],
            origin_coords=[0, 0, 0],
            origin_coords_unit=metadata['grid spacing unit'],
            reference_frame=f"Worm {metadata['location']}"
        )

        Image = build_colormap(data_path, file_name, ImagingVol, file_info, OpticalChannelRefs)
        nwb_file.add_imaging_plane(ImagingVol)
        print("Neuron ROIs processed and added to NWBFile.")  # Debug print

    print("Creating processed module...")  # Debug print
    processed_module = nwb_file.create_processing_module(
        name='Processed',
        description='Processed image data and metadata.'
    )

    # Discover and sort tiff files, build single .h5 file for iterator compatibility.
    calc_imaging_volume = build_gcamp(nwb_file, data_path, OpticalChannels, OpticalChannelRefs, main_device, metadata)

    build_nir(nwb_file, ImagingVol, h5_path)
    #video_center_plane, video_center_table, colormap_center_plane, colormap_center_table, NeuroPALImSeg = build_neuron_centers(
    #    data_path, ImagingVol, calc_imaging_volume)
    #build_activity(data_path, metadata, video_center_table)

    processed_module.add(Image)
    processed_module.add(NeuroPALImSeg)
    processed_module.add(OpticalChannelRefs)
    for eachBehavior in behavior:
        processed_module.add(eachBehavior)

    # specify the file path you want to save this NWB file to
    save_path = f"{data_path}\\{file_name}.nwb"
    io = NWBHDF5IO(save_path, mode='w')
    io.write(nwb_file)
    io.close()
    print("NWB file built and saved at:", save_path)  # Debug print

if __name__ == "__main__":
    iterate_folders()

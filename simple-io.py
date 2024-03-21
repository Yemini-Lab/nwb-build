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
        if os.path.isdir(item) and item not in existing_nwb_files and 'continuous' in item:
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
def discover_tiff_files(folder):
    # Initialize an empty list to store file paths
    tiff_files = []

    # Regular expression to match the naming scheme and extract metadata
    pattern = re.compile(r'worm(\d+)_run(\d+)_t(\d+)_(\w)\.tiff')

    # Iterate through all files in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.tiff'):
            match = pattern.match(filename)
            if match:
                # Extract metadata from filename
                worm_num, run_num, frame_num, channel = match.groups()

                # Append a tuple containing the full path and metadata to the list
                full_path = os.path.join(folder, filename)
                tiff_files.append((full_path, int(worm_num), int(run_num), int(frame_num), channel))

    # Sort the list based on the metadata (worm_num, run_num, frame_num, channel)
    tiff_files.sort(key=lambda x: (x[1], x[2], x[3], x[4]))

    # Return only the sorted file paths
    sorted_tiff_files = [x[0] for x in tiff_files]
    print(f"-FRAME COUNT: {len(sorted_tiff_files)}")

    return sorted_tiff_files


def h5_memory_mapper(folder_path):
    # Initialize variables
    output_file = f"{folder_path}\\video-array.h5"
    frame_pattern = re.compile(r"worm\d+_run\d+_t(\d+)_([RG])\.tiff")
    batch_size = 4
    max_workers = 1  # Number of threads

    # Check if .h5 file already exists
    if os.path.exists(output_file):
        print(f"-HDF5 FILE ALREADY EXISTS, SKIPPING BUILD.")
        return

    def process_file(tiff_file, h5_file, folder_path):
        match = frame_pattern.match(tiff_file)
        if match:
            frame_num = int(match.group(1))
            channel = 0 if match.group(2) == 'R' else 1
            tiff_data = imread(os.path.join(folder_path, tiff_file))
            h5_file['dataset'][frame_num, :, :, :, channel] = np.transpose(tiff_data, (2, 1, 0))

    # Get list of tiff files
    tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]
    print(f"Found {len(tiff_files)} tiff files.")

    # Find dimensions
    max_frame = -1
    sample_tiff = imread(os.path.join(folder_path, tiff_files[0]))
    z, y, x = sample_tiff.shape
    print(f"Sample tiff dimensions: {x} x {y} x {z}")

    for tiff_file in tiff_files:
        match = frame_pattern.match(tiff_file)
        if match:
            frame_num = int(match.group(1))
            max_frame = max(max_frame, frame_num)

    print(f"Max frame number: {max_frame}")

    # Create a memory-mapped .h5 file
    shape = (max_frame + 1, x, y, z, 2)
    print(f"Creating HDF5 file with shape {shape}")
    h5_file = h5py.File(output_file, 'w')
    h5_file.create_dataset('dataset', shape, dtype='uint16')

    try:
        # Populate the .h5 file in batches
        print("Populating the .h5 file...")
        for i in tqdm(range(0, len(tiff_files), batch_size)):
            batch_files = tiff_files[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_file, batch_files, [h5_file] * len(batch_files), [folder_path] * len(batch_files))
            gc.collect()
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


# Sort files based on timepoints
def sort_files(folder_path):
    files = os.listdir(folder_path)
    return sorted(files, key=lambda x: int(x.split('_t')[1].split('_')[0]))


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


def extract_mat_meta(full_path):
    # Load MATLAB info
    mat_file = next(f for f in os.listdir(full_path) if f.endswith('_info.mat'))
    mat_data = scipy.io.loadmat(os.path.join(full_path, mat_file))
    info = mat_data['info'][0, 0]

    def convert_mat_to_python(mat_array: Any) -> Any:
        """Convert MATLAB arrays to Python-native types."""
        if np.isscalar(mat_array):
            return mat_array.item()
        elif mat_array.size == 1:
            return mat_array.item().decode('utf-8') if isinstance(mat_array.item(), bytes) else mat_array.item()
        else:
            return mat_array.tolist()

    scan_rate = info['daq']['scanRate'][0][0][0][0]
    ai_sampling_rate = float(info['daq']['aiSampleRate'][0][0][0][0])

    return scan_rate, ai_sampling_rate


def build_gcamp(nwbfile, full_path, OptChannels, OpticalChannelRefs, device, location, pixel_sizes, unit):
    sorted_tiff_files = discover_tiff_files(os.path.join(full_path, 'FullRes'))
    h5_memory_mapper(os.path.join(full_path, 'FullRes'))
    scan_rate, ai_sampling_rate = extract_mat_meta(full_path)

    if sorted_tiff_files:
        with TiffFile(sorted_tiff_files[0]) as tif:
            numZ = len(tif.pages)  # Assuming each page is a Z slice
            image = cv2.imread(sorted_tiff_files[0])
            numX = image.shape[0]
            numY = image.shape[1]

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
        location=location,
        grid_spacing=pixel_sizes,
        grid_spacing_unit=unit,
        reference_frame=f"Worm {location}"
    )

    calcium_image_series = MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        description="GCaMP6s series images. Dimensions should be (t, x, y, z, C).",
        comments="",
        data=data,
        device=device,
        unit=unit,
        scan_line_rate=scan_rate,
        dimension=[numX, numY, numZ],
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
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
    # Extract relevant metadata
    session_description = file_data.get('session_description', 'Default description')
    identifier = file_data['identifier']
    try:
        session_start_time = datetime.strptime(file_data['dt'], '%Y%m%d %H:%M:%S').replace(
            tzinfo=tz.gettz("US/Eastern"))
    except:
        session_start_time = datetime.strptime(file_data['dt'], '%Y%m%d').replace(tzinfo=tz.gettz("US/Eastern"))
    lab = file_data.get('credit', {}).get('lab', 'Unknown Lab')
    institution = file_data.get('credit', {}).get('institution', 'Unknown Institution')
    related_publications = file_data.get('credit', {}).get('related_pubs', '')

    # Extract relevant subject metadata
    subject_metadata = file_data.get('subject', {})
    subject_id = file_data['identifier']

    date_str = subject_metadata.get('date_of_birth', file_data['dt']).split()[0]
    date_of_birth = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=tz.gettz("US/Eastern"))

    growth_stage = subject_metadata.get('growth_stage', 'Unknown')
    growth_stage_time = pd.Timedelta(hours=subject_metadata.get('growth_stage_time_h', 0),
                                     minutes=subject_metadata.get('growth_stage_time_m', 0)).isoformat()
    cultivation_temp = subject_metadata.get('cultivation_temp', 20.)
    description = subject_metadata.get('description', 'Default description')
    species = "http://purl.obolibrary.org/obo/NCBITaxon_6239"
    sex = subject_metadata.get('sex', 'O')
    strain = subject_metadata.get('strain', 'Unknown')

    # Create NWBFile object
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        lab=lab,
        institution=institution,
        related_publications=related_publications
    )

    # Create CElegansSubject object and add to NWBFile
    nwbfile.subject = CElegansSubject(
        subject_id=subject_id,
        date_of_birth=date_of_birth,
        growth_stage=growth_stage,
        growth_stage_time=growth_stage_time,
        cultivation_temp=float(cultivation_temp[:-1]),
        description=description,
        species=species,
        sex=sex,
        strain=strain
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


def build_channels():
    # Define channels and create OpticalChannelPlus objects
    channels = [("mTagBFP2", "", "405-488-50m"),
                ("CyOFP1", "", "488-594-30m"),
                ("GCaMP6s", "", "488-594-30m"),
                ("TagRFP-T", "", "594-637-70m"),
                ("mNeptune 2.5", "", "561-594-75m")]
    gcamp_channels = [("TagRFP-T", "", "594-637-70m"),
                      ("GCaMP6s", "", "488-594-30m")]

    OpticalChannels = []
    OptChanRefData = []

    for fluor, des, wave in channels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        OptChan = OpticalChannelPlus(
            name=fluor,
            description=des,
            excitation_lambda=excite,
            excitation_range=[excite - 1.5, excite + 1.5],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid
        )
        OpticalChannels.append(OptChan)
        OptChanRefData.append(wave)

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


def build_nwb(full_path, metadata):
    nwbfile, main_device = build_file(metadata)
    behavior = build_behavior(full_path, metadata)
    OpticalChannels, OpticalChannelRefs, gcamp_OpticalChannels, gcamp_OpticalChannelRefs = build_channels()
    location = metadata['colormap']['location']

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

    Image = build_colormap(full_path, ImagingVol, metadata, OpticalChannelRefs)
    nwbfile.add_imaging_plane(ImagingVol)

    processed_module = nwbfile.create_processing_module(
        name='Processed',
        description='Processed image data and metadata.'
    )

    # Discover and sort tiff files, build single .h5 file for iterator compatibility.
    calc_imaging_volume = build_gcamp(nwbfile, full_path, gcamp_OpticalChannels, gcamp_OpticalChannelRefs, main_device,
                                      location, extract_pixel_sizes(metadata['comments']), 'um')

    video_center_plane, video_center_table, colormap_center_plane, colormap_center_table, NeuroPALImSeg = build_neuron_centers(
        full_path, ImagingVol, calc_imaging_volume)
    build_activity(full_path, metadata, video_center_table)
    #build_mip(nwbfile, ImagingVol, full_path)

    processed_module.add(Image)
    processed_module.add(NeuroPALImSeg)
    processed_module.add(OpticalChannelRefs)
    processed_module.add(gcamp_OpticalChannelRefs)
    if behavior is not None:
        processed_module.add(behavior)

    # specify the file path you want to save this NWB file to
    save_path = full_path + ".nwb"
    io = NWBHDF5IO(save_path, mode='w')
    io.write(nwbfile)
    io.close()


if __name__ == "__main__":
    iterate_folders()

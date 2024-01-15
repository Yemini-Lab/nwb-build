import scipy.io
import configparser
import datetime
import gc
import json
import os
import shutil
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

def build_nir(nwbfile, video_path):
    with h5py.File(video_path, 'r') as hdf:
        data = hdf['img_nir'][:]
        timestamps = hdf['img_metadata']['img_timestamp'][:]

    nir_data = np.array(data)
    nir_data = np.transpose(data, axes=(0, 2, 1))  # Shape should be (T, X, Y)
    nir_data = nir_data[:, :, :] 
    hefty_data = H5DataIO(data=nir_data, compression=True)

    timesstamps = timestamps[::2]
    numtime = nir_data.shape[0]

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
        timestamps= timestamps[:numtime],
        #timestamps=list(range(hefty_data.shape[0])),
    )

    return nir_vol_series

def fix_nwb(nwb_path, export_path, data_path, file_name):

    with NWBHDF5IO(nwb_path, mode='r') as read_io:

        h5_path = f"{data_path}/{file_name}.h5"

        nwbfile = read_io.read()
        # ...  # modify nwbfile

        nwbfile.processing["BF_NIR"].data_interfaces.pop('BrightFieldNIR')

        nir_vol_series = build_nir(nwbfile, h5_path)

        nwbfile.processing["BF_NIR"].add(nir_vol_series)

        nwbfile.set_modified()  # this may be necessary if the modifications are changes to attributes

        with NWBHDF5IO(export_path, mode='w') as export_io:
            export_io.export(src_io=read_io, nwbfile=nwbfile)

if __name__ == "__main__":
    for folder in os.listdir('/mnt/flavell/data_raw'):

        if not folder[:2] == '20':
            continue

        failed_folders = ['2022-06-14-07', '2022-06-28-01', '2022-07-20-01', '2023-01-05-01', '2023-01-05-18', '2023-01-06-08', '2023-01-09-28', '2023-01-16-01', '2023-01-16-08', '2023-01-19-15', '2023-01-19-22', '2023-01-23-08']

        data_path = '/mnt/flavell/data_raw/'+ folder
        file_name = folder

        nwb_path = '/mnt/flavell/NWB_flavell_updated/' + folder +'.nwb'
        export_path = '/mnt/flavell/NWB_flavell_final/' + folder +'.nwb'

        if not folder in failed_folders:
            if not folder in ['2023-01-19-01', '2023-01-06-15']:
                if not os.path.exists(export_path):
                    os.rename(nwb_path, export_path)
            continue

        print(folder)
        
        #fix_nwb(nwb_path, export_path, data_path, file_name)

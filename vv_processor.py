import os
import re
import h5py
import simpleio
import nd2reader
from datetime import datetime
import pandas as pd
import time
from tqdm import tqdm
import json
from pathlib import Path
import scipy.io as sio
import numpy as np

map_root = Path("C:\\Users\\Yemini Laboratory\\Dropbox\\Male-Herm-Paper")
video_root = Path("C:\\Users\\Yemini Laboratory\\Northeastern University\\Seyedehmaedeh Seyedolmohadesin - "
                  "sexual_dimorphism_data\\")

f = open(f"{video_root}\\datasets.json")
datasets = json.load(f)

for eachFile in tqdm(datasets.keys(), desc="Processing files..."):
    experiment = datasets[eachFile]
    metadata = {
        'lab': 'Yemini Laboratory & Venkatachalam Laboratory',
        'institution': 'UMass Chan Medical School & Northeastern University',
        'cultivation_temp': '20C',
        'related_publications': '',

        'description': '',

        'date': str(experiment['date']),
        'sex': str(experiment['sex']),
        'animal_id': str(experiment['animal']),

        'video': {},
        'map': {}
    }

    metadata['identifier'] = f"{metadata['date']}_{metadata['sex'][0]}{metadata['animal_id']}"

    metadata['video']['path'] = video_root / metadata['date'] / metadata['sex'] / metadata['animal_id']
    video_json = json.load(open(f"{metadata['video']['path']}\\metadata.json"))
    metadata['video']['dims'] = {
        'nt': int(video_json['shape_t']),
        'nc': int(video_json['shape_c']),
        'nz': int(video_json['shape_z']),
        'ny': int(video_json['shape_y']),
        'nx': int(video_json['shape_x']),
        'bitDepth': str(video_json['dtype'])
    }

    metadata['map']['path'] = map_root / f"{metadata['sex']}s" / metadata['date'] / metadata['animal_id'] / "corrected_10percent"
    map_json = json.load(open(f"{metadata['map']['path']}\\metadata.json"))
    metadata['map']['dims'] = {
        'nt': int(map_json['shape_t']),
        'nc': int(map_json['shape_c']),
        'nz': int(map_json['shape_z']),
        'ny': int(map_json['shape_y']),
        'nx': int(map_json['shape_x']),
        'bitDepth': str(map_json['dtype'])
    }

    data = {
        'video': {
            'raw_path': metadata['video']['path'] / "data.h5",
            'proc_path': video_root / "processed_data" / metadata['date'] / metadata['sex'] / metadata['animal_id'],
            'annos': metadata['video']['path'] / "annotations.h5",
            'world': metadata['video']['path'] / "worldlines.h5"
        },
        'map': {
            'volume': metadata['map']['path'] / "data.mat",
            'neurons': metadata['map']['path'] / "data_ID.mat"
        }
    }

    data['map']['contents'] = sio.loadmat(data['map']['volume'])

    if metadata['sex'] == 'hermaphrodite':
        assert(data['map']['contents']['worm'][0][0][2][0] == 'XX')
    elif metadata['sex'] == 'male':
        assert(data['map']['contents']['worm'][0][0][2][0] == 'XO')

    assert metadata['map']['dims']['ny'] == np.shape(data['map']['contents']['data'])[0]
    assert metadata['map']['dims']['nx'] == np.shape(data['map']['contents']['data'])[1]
    assert metadata['map']['dims']['nz'] == np.shape(data['map']['contents']['data'])[2]
    assert metadata['map']['dims']['nc'] == np.shape(data['map']['contents']['data'])[3]

    metadata['location'] = data['map']['contents']['worm'][0][0][0][0]
    metadata['age'] = data['map']['contents']['worm'][0][0][1][0]
    metadata['strain'] = data['map']['contents']['worm'][0][0][3]
    metadata['notes'] = data['map']['contents']['worm'][0][0][4]
    metadata['map']['grid_spacing'] = data['map']['contents']['info'][0][0][1][0]
    metadata['map']['RGBW'] = data['map']['contents']['prefs'][0][0]
    metadata['npal_version'] = data['map']['contents']['version'][0][0]

    data['map']['contents'] = data['map']['contents']['data']

    package = {'metadata': metadata, 'data': data}
    nwb_file, main_device, nir_device = simpleio.build_file(package)
    simpleio.build_nwb(nwb_file, package, main_device, nir_device)




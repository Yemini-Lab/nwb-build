import os
import re
import h5py
import simpleio
import nd2reader
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import time

# Additional import for tqdm
from tqdm import tqdm

lab = 'Flavell Laboratory'
institution = 'MIT'
grid_spacing = [0.54, 0.54, 0.54]
cultivation_temperature = '22C'
volume_rate = '1.7 Hz'
related_publications = 'https://doi.org/10.1016/j.cell.2023.07.035'

devices = {
    'vol': {
        'name': 'Spinning Disk Confocal',
        'description': 'The light path used to image GCaMP, mNeptune, and the fluorophores in NeuroPAL at single cell '
                       'resolution is an Andor spinning disk confocal system with Nikon ECLIPSE Ti microscope. Light '
                       'supplied from a 150 mW 488 nm laser, 50 mW 560 nm laser, 100 mW 405 nm laser, or 140 mW 637 '
                       'nm laser passes through a 5000 rpm Yokogawa CSU-X1 spinning disk unit with a Borealis upgrade '
                       '(with a dual-camera configuration). A 40x water immersion objective (CFI APO LWD 40X WI 1.15 '
                       'NA LAMBDA S, Nikon) with an objective piezo (P-726 PIFOC, Physik Instrumente (PI)) was used '
                       'to image the volume of the worm’s head (a Newport NP0140SG objective piezo was used in a '
                       'subset of the recordings). A custom quad dichroic mirror directed light emitted from the '
                       'specimen to two separate sCMOS cameras (Zyla 4.2 PLUS sCMOS, Andor), which had in-line '
                       'emission filters (525/50 for GcaMP/GFP, and 610 longpass for mNeptune2.5; NeuroPAL filters '
                       'described below). Data was collected at 3 × 3 binning in a 322 × 210 region of interest in '
                       'the center of the field of view, with 80 z planes collected at a spacing of 0.54 um. This '
                       'resulted in a volume rate of 1.7 Hz (1.4 Hz for the datasets acquired with the Newport piezo).',
        'manufacturer': 'Andor, Nikon'
        },
    'nir': {
        'name': 'NIR Config',
        'description': 'The light path used to image behavior was in a reflected brightfield (NIR) configuration. '
                       'Light supplied by an 850-nm LED (M850L3, Thorlabs) was collimated and passed through an '
                       '850/10 bandpass filter (FBH850-10, Thorlabs). Illumination light was reflected towards the '
                       'sample by a half mirror and was focused on the sample through a 10x objective (CFI Plan Fluor '
                       '10x, Nikon). The image from the sample passed through the half mirror and was filtered by '
                       'another 850-nm bandpass filter of the same model. The image was captured by a CMOS camera ('
                       'BFS-U3-28S5M-C, FLIR).',
        'manufacturer': 'Nikon'
        },
}
#data_path = '/Volumes/FlavellNP/data_raw/'
data_path = '/mnt/flavell/data_raw/'
#data_path = '/Users/danielysprague/foco_lab/data/Flavell_example/'
directory = '2022-06-14-01'
#metadata_file = pd.read_csv('/Users/danielysprague/foco_lab/data/Flavell_example/flavell_data_new.csv')
#metadata_file = pd.read_excel('/Volumes/FlavellNP/data_raw/flavell_data.xlsx')
#metadata_file = pd.read_csv('/home/jackbo/NWB-conversion/flavell_data.csv')
metadata_file = pd.read_csv('/mnt/flavell/data_raw/flavell_data.csv')
#metadata_file = pd.read_csv('/Volumes/FlavellNP/data_raw/flavell_data.csv')

def conv_file(file_name, file_path, file_extension):
    print(f"Processing file: {file_name}{file_extension} in {file_path}")  # Debug print
    file_info = {}
    file_info['lab'] = lab
    file_info['institution'] = institution
    file_info['cultivation_temp'] = cultivation_temperature
    file_info['related_publications'] = related_publications
    file_info['devices'] = devices
    file_info['path'] = file_path

    match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)", file_name)
    if match:
        file_info['date'] = match.group(1)
        file_info['subj_no'] = match.group(2)
    else:
        print("No match found in filename regex.")  # Debug print
        return
    

    file_info['metadata'] = extract_subject_data(metadata_file, file_info['date'], int(file_info['subj_no']))
    file_info['identifier'] = file_info['date'] +'-'+ str(file_info['subj_no'])
    file_info['metadata'][0]['date'] = datetime.strptime(file_info['metadata'][0]['date'], '%Y-%m-%d')

    nwb_file, main_device, nir_device = simpleio.build_file(file_info)

    simpleio.build_nwb(nwb_file, file_info, file_info['subj_no'], main_device, nir_device)
    file_info['x'] = nd2reader.ND2Reader(file_info['path']).metadata['width']
    file_info['y'] = nd2reader.ND2Reader(file_info['path']).metadata['height']
    file_info['num_frames'] = nd2reader.ND2Reader(file_info['path']).metadata['num_frames']
    file_info['c'] = len(nd2reader.ND2Reader(file_info['path']).metadata['channels'])

    print(f"Completed processing for {file_name}{file_extension}")  # Debug print


def extract_subject_data(df, date, run):
    run = str(run)
    df = df[df['date'] == date]
    animal = df.loc[df['run'] == run, 'animal'].values[0]
  
    subject_data = df[df['animal'] == animal]
    subject_data = subject_data.reset_index()
    result = {}
    for index, row in subject_data.iterrows():
        result[index] = row.to_dict()
    return result

def process_file(directory):
    file_name = directory
    file_extension = '.nd2'
    full_path = data_path + directory + '/' +file_name +file_extension
    conv_file(file_name, full_path, file_extension)

def process_directory(directory):
    entries = os.listdir(directory)
    for entry in tqdm(entries, desc="Processing Directory"):  # Loading bar
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry)
            if file_extension == '.nd2':
                conv_file(file_name, full_path, file_extension)

'''
for folder in tqdm(os.listdir(data_path)):
    if folder[0:2]!= '20':
        continue

    if folder in ['2023-01-19-01', '2023-01-06-15']:
        continue
    #if os.path.exists(data_path + 'NWB_NP_flavell/'+folder + '.nwb'):
    #    continue
    print(folder)
    t1 = time.time()
    process_file(folder)
    t2 = time.time()
    print(str(t2-t1))
'''
process_file(directory)

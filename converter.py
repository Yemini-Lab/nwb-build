import os
import re
import h5py
import simpleio
import nd2reader
import pandas as pd

lab = 'Flavell Laboratory'
institution = 'MIT'
related_publications = ''
metadata_file = pd.read_csv('')

def conv_file(file_name, file_path, file_extension):
    file_info = {}
    file_info['lab'] = lab
    file_info['institution'] = institution
    file_info['related_publications'] = related_publications
    file_info['path'] = f"{file_name}{file_extension}"
    
    # Extract date and subject number from the file name
    match = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)", file_name)

    if match:
        file_info['date'] = match.group(1)
        file_info['subj_no'] = match.group(2)
    else:
        return
    
    file_info['metadata'] = extract_subject_data(metadata_file, f"{file_info['date']}-{file_info['subj_no']}")
    nwb_file, main_device = simpleio.build_file(file_info['metadata'])

    for eachRun in range(len(file_info['metadata'].keys())):
        simpleio.build_nwb(nwb_file, file_info['path'], file_info['metadata'][eachRun], main_device)

        if file_extension == '.h5':
            with h5py.File(file_path, 'r') as hdf:
                dataset = 'img_metadata'
                if dataset in hdf:
                    data = hdf[dataset]['img_timestamp'][:]
                    print(len(data))
                    print(data)
                else:
                    print(f"Dataset {dataset} not found in the file.")
                file_info['x'] = hdf[dataset]['img_nir'].shape[1]
                file_info['y'] = hdf[dataset]['img_nir'].shape[2]
                file_info['num_frames'] = len(hdf[dataset]['img_timestamp'][:])
                file_info['c'] = len(nd2reader.ND2Reader(file_info['path']).metadata['channels'])
        elif file_extension == '.nd2':
            file_info['x'] = nd2reader.ND2Reader(file_info['path']).metadata['width']
            file_info['y'] = nd2reader.ND2Reader(file_info['path']).metadata['height']
            file_info['num_frames'] = nd2reader.ND2Reader(file_info['path']).metadata['num_frames']
            file_info['c'] = len(nd2reader.ND2Reader(file_info['path']).metadata['channels'])
        else:
            print('else')


def extract_subject_data(df, subject_name):
    """
    Extract data for a given subject from the dataframe.
    """
    # Filter the dataframe for the given subject
    subject_data = df[df['subject'] == subject_name]

    # Convert each row to a dictionary
    result = {}
    for index, row in subject_data.iterrows():
        # Using the index as the key for each subject's row
        result[index] = row.to_dict()

    return result


def process_directory(directory):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)

        # Check if the entry is a file and not a directory
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry)

            # Check for specific file extensions
            if file_extension in ['.h5', '.nd2']:
                conv_file(file_name, directory, file_extension)


process_directory('.')

################################
# A collection of utility functions for nested room id project
# SECTIONS:
# GENERAL UTILITIES
# DATA MANIPULATION, CONFIGURATION & SORTING
# IMAGE PROCESSING
################################

#############################################
# import required libraries
#############################################

import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
from scipy import misc
from datetime import datetime as dt


# (Now) unused imports
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
#import seaborn as sns
#sns.set()
#from keras.preprocessing import image
#from keras.applications.inception_v3 import preprocess_input


###############################################
# GENERAL UTILITIES
###############################################


def timestamp():
    '''
    Provide a 'now' timestamp formatted as string to append to file &
    directory names when naming and saving

    returns
    formatted string timestamp ('%Y-%m-%d_%H-%M-%S') for current time
    '''
    return dt.now().strftime('%Y-%m-%d_%H-%M-%S')



def pywalker_list(path, file_ext='.jpg'):
    '''
    returns a list of 'path/to/*.'file_ext' '  in input path for files
    matching *.file_ext

    args
    path - str - path to root directory for pywalker_list
    file_ext - str - default '.jpg' - extension of file type
        to be indexed

    returns
    list of filepaths (incl filename) to all files in root filepath
        matching file_ext
    '''
    file_list = []
    for root, dirs, files in os.walk(path):
        for file_ in files:
            filename = os.fsdecode(file_)
            if filename.endswith(file_ext):
                file_list.append( os.path.join(root, file_) )

    return file_list


############################################
# DATA MANIPULATION, CONFIGURATION & SORTING #
############################################


def configure_base_directory(path, rmv_not_pic=True):
    '''
    Configures an unzipped directory (as provided) into
    format & structure for further work

    args
    path - str - path to directory as unzipped from .tar.bz2
        (likely to be similar to
         './data/tagged-property-images-20180304.tar.bz2')
    rmv_not_pic - bool - default True - remove all non *.jpg files

    action:
    directory remaned as ./data/base_data_NEW/
    ./interior/misc renamed to misc_int
    ./interior/misc renamed to misc_ext
    all 'room' sub dirs move up a level, and 'interior' & 'exterior' dirs
    removed
    (if rmv_not_pic is True (default)) - removes all non *.jpg files

    '''
    # Moves source directory to ./data/base_data_NEW
    # Renames misc directories as misc_ext and misc_int
    # Creates 'uncertain' and 'graphic' directories
    print('Moving directory')
    os.rename(path, './data/base_data_NEW')
    print('Renaming misc directories')
    os.rename('./data/base_data_NEW/interior/misc', './data/base_data_NEW/interior/misc_int')
    os.rename('./data/base_data_NEW/exterior/misc', './data/base_data_NEW/exterior/misc_ext')
    print("Creating 'uncertain' & 'graphic' directory")
    os.mkdir('./data/base_data_NEW/uncertain')
    os.mkdir('./data/base_data_NEW/graphic')

    # moves room / scene directories out of ./interior and ./exterior
    # subdirs, and places all at same level under ./base_data_NEW
    print('Move room directories out of /interior & /exterior')
    for root, dirs, files in os.walk('./data/base_data_NEW'):
        root_list = root.split('/')
        parent_dir = root_list[-2:-1][0]
        test = ['exterior', 'interior']
        if parent_dir in test:
            root_list.pop(-2)
            new_path = os.path.join(*root_list)
            os.rename(root, new_path)

    # pauses (to ensure job completion above)
    # then deletes relict interior and exterior directories
    time.sleep(2)
    print('Deleting /interior and /exterior')
    shutil.rmtree('./data/base_data_NEW/interior/')
    shutil.rmtree('./data/base_data_NEW/exterior/')

    # if rmv_not_pic is True, then delete all non .jpg files
    if rmv_not_pic:
        print('Deleting non *.jpg files')
        count = 0
        for root, dirs, files in os.walk('./data/base_data_NEW'):
            for file_ in files:
                filename = os.fsdecode(file_)
                if not filename.endswith('.jpg'):
                    count += 1
                    #print('deleting file: {}'.format(file_))
                    os.remove(os.path.join(root, file_))
        print('Removed {} non .jpg files'.format(count))

    return


def promote_new_base_dir(path, old_dir_path='./data/base_data_OLD'):
    '''
    Promotes a (newly created) /base_data_NEW/ directory to be main
    /base_data/ directory for onward processing.
    Renames any existing .data/base_data as base_data_OLD to avoid
    accidental deletion

    Args
    path - str - path to the directory to be 'promoted' to /base_data

    old_dir_path - str - default './data/base_data_OLD'
        path to rename any existing base_data directory
    '''
    if os.path.exists('./data/base_data'):
        os.rename('./data/base_data', './data/base_data_OLD')
    os.rename(path, './data/base_data')
    return


def filename_from_path(row):
    "returns filename from filepath for use with df.apply()"
    return row['image_path'].split('/')[-1:][0]


def id_from_filename(row):
    "returns id from filename, for use with df.apply()"
    return row['filename'].split('.')[0]


def room_from_path(row):
    "returns room (directory name) from filepath, for  use with df.apply()"
    return row['image_path'].split('/')[-2]



def build_catalogue(path):
    '''
    Builds a catalogue (as pandas df) of *.jpg files in path

    Args
    path - str - path to root directory to catalogue

    Returns pandas df with columns:
    - image_path: full path to image file
    - filename: filename (from image path)
    - id: filename before '.jpg' file extension
    - room: room classification of image, (from directory allocation)
    '''
    # creates empty df, then uses pywalker_list to create 'image_path'
    # column data
    data_cat = pd.DataFrame()
    data_cat['image_path'] = pywalker_list(path)

    # uses functions to extract filename, id and room from 'image_path'
    # field
    data_cat['filename'] = data_cat.apply(filename_from_path, axis=1)
    data_cat['id'] = data_cat.apply(id_from_filename, axis=1)
    data_cat['room'] = data_cat.apply(room_from_path, axis=1)

    # reorders columns
    data_cat = data_cat[['id', 'room', 'filename', 'image_path']]

    return data_cat



def create_test_dirs(source, destination):
    """
    Creates directories for test data
    Sub-directory structure (classes) in source is replicated
    in test directory

    args:
    source - path to data source directory (str)
    destination - path to destination directory (str)
    """
    # defines train and validation directory names & paths
    test_dir = 'test_data'
    test_path = os.path.join(destination, test_dir)

    # creates destination directory (if not exist)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # walks through source subdirectories, and creates copy
    # in training and validation directories
    for dirpath, dirs, files in os.walk(source):
            for subdir in dirs:
                newpath_test = os.path.join(test_path, subdir)
                #print(newpath_test)
                if not os.path.exists(newpath_test):
                    os.mkdir(newpath_test)



def data_model_test_split(source, destination, test_split=0.25, filetype='.jpg'):
    """
    Walks through files in directory, and randomly moves a proportion to
    test set.
    Moves file to relevant test dir as appropriate

    args:
    source - source directory path (str)
    destination - destination directory path (str)
    test_split - test proportion in range 0-1 (default=0.25)
    """
    # defines paths to destination test directories
    test_path = destination

    # setsup file counters
    file_count_m = 0
    file_count_test = 0
    file_count_model = 0

    #loops through files in source
    for dirpath, dirs, files in os.walk(source):
            for file_ in files:
                # decodes filename, and checks if it is desired type
                filename = os.fsdecode(file_)
                if filename.endswith(filetype):
                    file_count_m += 1
                    # creates path for file as 'source'
                    filepath_source = os.path.join(dirpath, file_)
                    subdir = dirpath.split('/')[-1:][0]   # collects subdir(class) name as str
                    # randomly allocates to test or model set based on random number
                    if np.random.rand(1) <= test_split:
                        # test sample - defines path for file as destination, and moves file
                        file_count_test += 1
                        filepath_dest = os.path.join(test_path, subdir, file_)
                        #print(filepath_dest)
                        shutil.move(filepath_source, filepath_dest)
                    else:
                        # training sample - does nothing
                        file_count_model += 1


    print('{} files found: {} allocated to test and {} to modelling'
            .format(file_count_m, file_count_test, file_count_model))



def create_train_and_val_dirs(source, destination):
    """
    Creates training and validation directories for train/val split files
    Sub-directory structure (classes) in source is replicated
    in both training and validation directories

    args:
    source - path to data source directory (str)
    destination - path to destination directory (str)
    """
    # defines train and validation directory names & paths
    train_dir = 'training_dir'
    val_dir = 'validation_dir'
    train_path = os.path.join(destination, train_dir)
    val_path = os.path.join(destination, val_dir)

    # creates destination directory (if not exist)
    if not os.path.exists(destination):
        os.mkdir(destination)

    # creates training and validation directories (if not exist)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    # walks through source subdirectories, and creates copy
    # in training and validation directories
    for dirpath, dirs, files in os.walk(source):
            for subdir in dirs:
                newpath_t = os.path.join(train_path, subdir)
                newpath_v = os.path.join(val_path, subdir)
                if not os.path.exists(newpath_t):
                    os.mkdir(newpath_t)
                if not os.path.exists(newpath_v):
                    os.mkdir(newpath_v)


def data_train_val_split(source, destination, train_dir='training_dir',
                         val_dir = 'validation_dir',
                         filetype='.jpg', val_split=0.3):
    """
    Walks through files in directory, and randomly assigns & copies to
    training or validation set.
    Copies file to destination /train or /val as appropriate

    args:
    source - source directory path (str)
    destination - destination directory path (str)
    train_dir - training directory name in destination (default='training_dir')
    val_dir - val directory name in destination (default='training_dir')
    filetype - file extension of filetypes to consider (default='.jpg')
    val_split - validation proportion in range 0-1 (default=0.3)
    """
    # defines paths to destination train and val directories
    dest_train_path = os.path.join(destination, train_dir)
    dest_val_path = os.path.join(destination, val_dir)
    # setsup file counters
    file_count_m = 0
    file_count_t = 0
    file_count_v = 0

    #loops through files in source
    for dirpath, dirs, files in os.walk(source):
            for file_ in files:
                # decodes filename, and checks if it is desired type
                filename = os.fsdecode(file_)
                if filename.endswith(filetype):
                    file_count_m += 1
                    # creates path for file as 'source'
                    filepath_source = os.path.join(dirpath, file_)
                    subdir = dirpath.split('/')[-1:][0]
                    # collects subdir(class) name as str
                    # randomly allocates to train or val based on val_split proportion
                    if np.random.rand(1) < val_split:
                        # validation sample - defines path for file as destination, and copies file
                        file_count_v += 1
                        filepath_dest = os.path.join(dest_val_path, subdir, file_)
                        shutil.move(filepath_source, filepath_dest)
                    else:
                        # training sample - defines path for file as destination, and copies file
                        file_count_t += 1
                        filepath_dest = os.path.join(dest_train_path, subdir, file_)
                        #print('\nTraining sample: {}\nTarget path: {}'.format(filepath_source, filepath_dest))
                        shutil.move(filepath_source, filepath_dest)

    print('{} files copied: {} training and {} validation'
            .format(file_count_m, file_count_t, file_count_v))






##############################################
# IMAGE PROCESSING AND PLOTTING #
##############################################

def read_img(img, size=(299,299)):
    """Read and resize image.
    # Arguments
        img -str - path for image
        size - 2-tuple - default (299,299) - size for output image
    # Returns
        img (for plotting or prediction)
    """
    img = misc.imread(img)
    img = cv2.resize(img, (size))
    return img

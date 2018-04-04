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

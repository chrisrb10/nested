################################
# A collection of utility functions for nested room id project
# IMAGE PROCESSING AND PLOTTING
################################

#############################################
# import required libraries
#############################################


#import time
import cv2
import os
import numpy as np
import pandas as pd
from scipy import misc
from datetime import datetime as dt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns
sns.set()

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



def image_grid(data, image_path, image_size='m', figsize=(22,11),
               annotate=False, save=True, fill_grid=True,
               f_stem='image_grid_', fig_title='Figure Title',
               label_fn=None, label_info=None,
               output_dir='working_data/comp_images/'):
    '''
    Image grid function.
    Takes information from a pandas df (inclunding image path and label
        information), and builds an image grid of images +/- labels

    args:
    data - pandas dataframe - a pandas dataframe, containing image path and
        label content information
    image_path - str - default = 'image_path' - source dataframe column with
        image_path information

    kwargs:
    image_size - 's','m','l','xl', 'xxl' - image size in grid, sets image grid
        dimensions
    figsize - tuple -  output figure size
    annotate - bool (default=False) - whether to annotate (call label_fn)
        images in grid
    save - bool (default=True) - whether to save image_grid and csv of
        source files
    fill_grid - bool (default=True) - completely fill the image grid, or allow
        blank cells if too few images
    f_stem - str - stem for filename of save files (before timestamp)
    fig_title - str - Image grid title

    label_fn - fn - labelling function to call on images in grid
    label_info - dict -  dict of params as required by relevant label function

    output_dir - str - (default = 'working_data/comp_images/') - path to
        directory for saving output

    TO DO:


    '''

    # sets size of image grid and labels (fontsize)
    if image_size == 's':
        grid_size = (5,10)
        label_size = 8
    elif image_size == 'm':
        grid_size = (4,8)
        label_size = 10
    elif image_size == 'l':
        grid_size = (3,6)
        label_size = 12
    elif image_size == 'xl':
        grid_size = (2,4)
        label_size = 14
    elif image_size == 'xxl':
        grid_size = (1,2)
        label_size = 16


    # sets up image grid and turns off axis on all axes
    fig = plt.figure(1, figsize=figsize)

    grid = ImageGrid(fig, 111, nrows_ncols=grid_size, axes_pad=0.05)
    for ax in grid:
        ax.set_axis_off()

    # Counfigures 'source' df for image_grid
    # If input df (data) contains MORE images than cells in grid, then
    # use df.sample() to randomly select the required number of images
    # If there are LESS images in input df (data) than cells in grid, then
    # consider 'fill-grid' param:
    # If fill-grid=True, then use df.sample() with replace=True to fill
    # up the grid.
    # If fill_grid=False, then just use the data as source, and output partial
    # grid (with all images in input df shown once).
    # When input df(data) equals cells in grid, then show each image once.
    if data.shape[0] > grid.ngrids:
         source = data.sample(grid.ngrids, replace=False)
    elif data.shape[0] < grid.ngrids:
        if fill_grid:
            source = data.sample(grid.ngrids, replace=True)
        else:
            source=data
    elif data.shape[0] == grid.ngrids:
        source = data


    # set grid super title, and use tight_layout to fill space appropriately
    fig.suptitle(fig_title, fontsize=18)
    fig.tight_layout(rect=[0,0, 1, 0.95])

    #iterates through rows of df, plots image (from image_path) and
    # (if annotate=True) calls appropriate label function (label_fn)
    for i, (index, row) in enumerate(source.iterrows()):
        if i > grid.ngrids-1: break
        ax = grid[i]
        img = read_img(row[image_path])
        ax.imshow(img)
        if annotate:
            label_fn(ax, row, label_info, label_size)

    # if save=True, then save composed grid as *.png, and source
    # (list if images in this grid) as .csv in subdir
    # use of timestamp enables matching of grid image and source data of
    # images it contains
    # timestamp also ensures no overwriting of files, and allow multiple
    # successive calls on this function (e.g. from large image list)
    # to create a number of randomly sampled grids
    if save:
        f_timestamp = timestamp()
        # checks if save dirs exist, and creates them if not
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_dir+'csv_files/'):
            os.mkdir(output_dir+'csv_files/')
        # saves figure and corresponding .csv list of images
        fig.savefig(output_dir+f_stem+f_timestamp+'.png')
        source.to_csv(output_dir+'csv_files/'+f_stem+f_timestamp+'.csv')


def label_before_after(ax, row, label_info, label_size):
    '''
    Annotate images with original and reclassified image labels
    For use with image_grid function

    Args
    ax - axes object - current plotting axes object
    row - df row  - current df row
    label_info - dict - dict of params passed through image_grid
    label_size - int - fontsize (from image_grid)
    '''
    # OLD approach (redacted) - reading from dict with no defaults
    #before = label_info['before']
    #after = label_info['after']

    # BETTER approach - reading from dict if present, setting default value
    # if not.
    # before and after are column names to retrive 'original' and 'reclassified'
    # image label for plotting on image in grid
    before = label_info.get('before', 'before')
    after = label_info.get('after', 'after')

    # creates horizontal background bars (default colours) with
    # text for original label and reclassified label
    # text extracted from df row provided using column labels extracted from
    # label_info dict (or defaults)
    ax.barh(255, 298, 24, left=0, alpha=0.8)
    ax.text(5, 255,'Original: {}'.format(row[before]), color='white',
    fontsize=label_size, va='center')

    ax.barh(280, 298, 24, left=0, alpha=0.8)
    ax.text(5, 281,'Re-labelled: {}'.format(row[after]), color='white',
    fontsize=label_size, va='center')


def label_probs(ax, row, label_info, label_size):
    '''
    Annotate images with predicted image labels and probabilities
    on bars scaled to probability
    For use with image_grid function

    Args
    ax - axes object - current plotting axes object
    row - df row  - current df row
    label_info - dict - dict of params passed through image_grid
    label_size - int - fontsize (from image_grid)
    '''
    # extracting params from label_info dict (or setting defaults)
    # num_labels - int - default=2 - number of labels to show (1-3)
    num_labels = label_info.get('num_labels', 2)
    # show_actual - bool - default=False - show the 'actual' label text
    show_actual = label_info.get('show_actual', False)
    # label1-3 - extract df column names for labels 1,2,3 data
    label1 = label_info.get('label1', 'label1')
    label2 = label_info.get('label2', 'label2')
    label3 = label_info.get('label3', 'label3')
    # label1,2,3_P - extract df column names for labels1,2,3 probabilities
    label1_P = label_info.get('label1_P', 'label1_P')
    label2_P = label_info.get('label2_P', 'label2_P')
    label3_P = label_info.get('label3_P', 'label3_P')
    # room - extract df column label for 'actual' room label
    room = label_info.get('room', 'room')
    # threshold plotting
    threshold = label_info.get('threshold', 0)

    # plot label 1 & prob on horizontal bar (scaled to label_P)
    if num_labels > 0:
        if row[label1_P] >= threshold:
            ax.barh(230,(298*row[label1_P]),24, left=0, alpha=0.8)
            ax.text(5, 230,'{}: {:.3f}'.format(row[label1], row[label1_P]),
                    color='white', fontsize=label_size, va='center')

    # plot label 2 & prob on horizontal bar (scaled to label_P)
    if num_labels > 1:
        if row[label2_P] >= threshold:
            ax.barh(255,(298*row[label2_P]),24, left=0, alpha=0.8)
            ax.text(5, 255,'{}: {:.3f}'.format(row[label2], row[label2_P]),
                    color='white', fontsize=label_size, va='center')

    # plot label 3 & prob on horizontal bar (scaled to label_P)
    if num_labels > 2:
        if row[label3_P] >= threshold:
            ax.barh(280,(298*row[label3_P]),24, left=0, alpha=0.8)
            ax.text(5, 280,'{}: {:.3f}'.format(row[label3], row[label3_P]),
                    color='white', fontsize=label_size, va='center')

    # show actual label (if show_actual=True)
    if show_actual:
        ax.text(5, 20, row['room'], color='white',
                fontsize=label_size, backgroundcolor='gray', alpha=0.8)


def normalise_cm(cm):
    '''
    Normalises a confusion matrix
    '''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return(cm)


def plot_confusion_matrix(conf_mat, labels, fig, ax,
                          annot=True, fontsize=16,
                          normalise=True):
    '''
    plots confusion matrix heatmap
    need to add: titling, axis & tick fontsize
    '''

    if normalise:
        conf_mat = normalise_cm(conf_mat)
        vmin = 0
        vmax = 1
        fmt = '.2f'
    else:
        vmin = None
        vmax = None
        fmt = 'd'

    cm_df = pd.DataFrame(conf_mat, index=labels, columns=labels)

    ax = sns.heatmap(cm_df, ax=ax, annot=annot,
                    annot_kws={'fontsize':fontsize}, fmt=fmt,
                    vmin=vmin, vmax=vmax, cmap='YlGnBu')

    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(),
                            rotation=45, ha='right', fontsize=14)
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(),
                            rotation=0, ha='right', fontsize=14)

    ax.set_xlabel('Predicted Room', fontsize=16)
    ax.set_ylabel('Actual Room', fontsize=16)
    fig.suptitle('Confusion Matrix Heatmap - Interior Room Classifier',
                fontsize=18)
    fig.tight_layout(rect=[0,0, 1, 0.95])

    return

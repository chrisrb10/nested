################################
# Prediction and prediction data manipulation for nested project
################################

#############################################
# import required libraries
#############################################

import numpy as np
import pandas as pd
from scipy import misc

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

import nested_utilities as nutil

##########################################################################
# PREDICTING ######
##########################################################################


def prep_for_predict(img):
    '''
    Prepares an img (from read_img) for prediction using a keras (Inception)
    model.
    Uses keras libraries to convert to array, expand dimensions, and
    preprocess input (normalise colours) ready for CNN prediction
    '''
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict(model, img):
    '''
    Makes label prediction for img with model
    '''
    x = prep_for_predict(img)
    pred = model.predict(x)
    return pred



def predict_catalogue(data, model, image_path='image_path'):
    '''
    Makes label predictions for all image_path in data (df) using model

    args
    data - df - input df of images for prediction
    model - keras model to be used for predictions
    image_path (default='image_path') - column name in data for image filepaths

    '''
    # setup blank array for prediction data, getting number of labels
    # from model.output_shape
    num_labels = model.output_shape[1]
    pred_array = np.empty((0,num_labels))

    # setup counter (for commenting on progress)
    count = 0

    # interate through rows in data df
    print('Starting prediction for {} images'.format(data.shape[0]))
    for index, row in data.iterrows():
        count +=1
        if count % 100 == 0:
            # prints output every hundred rows
            print('processing row {}'.format(count))
        # reads image from image path
        img = nutil.read_img(row[image_path])
        # makes prediction using predict function
        pred = predict(model, img)
        # appends prediction data to pred_array
        pred_array = np.append(pred_array, pred, axis=0)

    print('Predictions complete. {} image predictions made'.format(count))
    return pred_array



def pred_to_df(pred_array, columns):
    '''
    Creates a df from an array of predictions and a set of column (label) names

    args
    pred_array - np array - output from predict_catalogue
    columns - list of str - labels corresponding to model output pred_array
    '''
    if len(columns) != pred_array.shape[1]:
        print('ERROR - # column names does not match # of labels in prediction')
        return

    # creates pred_df from pred_array data, with labelled columns
    pred_df = pd.DataFrame(pred_array, columns=columns)
    return pred_df


def probs_to_df(pred_df, num_labels=3):
    '''
    Extract the top 'num_labels' (3) probabilities for each image
    takes pred_df data .values to convert into a np.array
    calls np.sort on this array, and takes the num_labels (3) largest
    values in each row ([:,-(num_labels):])
    puts the result into a new df (probs_df), note column labelling is in reverse
    probability order (np.sort returns in ascending order)
    re-orders df columns to the more rational order (1,2,3)

    returns
    a df with top num_labels names and associated proabiliies for each
    image in pred_df
    '''


    # creates column labels for label probs - sort into reverse order
    columns = []
    for i in range(num_labels):
        columns.append('label'+str(i+1)+'_P')
    columns.sort(reverse=True)

    # selects 'num_labels' highest probabilities from pred_df
    # writes into probs_df, and sorts to label1 first
    probs_df = pd.DataFrame(np.sort(pred_df.values)[:,-(num_labels):],
                            columns=columns)
    probs_df = probs_df.reindex_axis(sorted(probs_df.columns), axis=1)

    return probs_df


def labels_to_df(pred_df, num_labels=3):
    '''
    Extracts the labels corresponding to the top num_labels(3) probabilities in
    probs_df.
    Calls np.argsort on same pred_df data (as np.array) as probs_df extraction.
    Argsort returns indices of 3 largest values in each row
    Iterate through each row, and use the indices to extract the corresponding
    column label from the list of pred_df column names (pred_df.columns)
    Creates a dictionary (e.g. {'label1':'bathroom', 'label2':'kitchen, etc)
    for each row, and appends to a list
    Thes converts list of dictionaries (all with matching keys) to a df

    (NOTE this could have been done using df.append - but that is substantially
    slower, as pd.concat (and therefore pd.append) make a
    full copy of he data each time)

    returns
    a dataframe of the top 3 labels (in order) for each image
    '''


    label_idx = np.argsort(pred_df.values)[:,-(num_labels):]

    rows_list = []

    for row in label_idx:
        row_dict ={}
        key_list = []
        val_list = []
        for i in range(num_labels):
            i = i+1
            key_list.append('label'+str(i))
            val_list.append(pred_df.columns[row[-i]])
        row_dict = dict(zip(key_list, val_list))
        rows_list.append(row_dict)

    labels_df = pd.DataFrame(rows_list)
    return labels_df


def combine_labels_probs_df(labels_df, probs_df):
    '''
    Combine the labels_df and probs_df dataframes to give
    a single df with top 3 labels and associated probabilities
    '''
    return pd.concat([labels_df, probs_df], axis=1)



def get_labels_and_probs(pred_df, num_labels=3):
    '''
    Sequentially runs the 3 functions to extract top n (num_labels)
    labels and associated probabilities from prediction data and combine
    into a single df

    returns
    dataframe with top n labels and top n probabilities
    '''

    probs_df = probs_to_df(pred_df, num_labels=num_labels)
    labels_df = labels_to_df(pred_df, num_labels=num_labels)
    pred_labels = combine_labels_probs_df(labels_df, probs_df)
    return pred_labels

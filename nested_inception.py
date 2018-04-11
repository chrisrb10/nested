#############################################################
# Code to retrain an InceptionV3 model for nested room image classifier
# Use function call under "if __name__=='__main__':" at end to configure
# training run parameters for a specific run
#############################################################


# IMPORT REQUIRED LIBRARIES AND FUNCTIONS
import os
import sys
import glob
import pandas as pd
from datetime import datetime as dt

# import required Keras models & functions
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger

# provides an 'hourstamp' (no minutes or seconds) for naming directory to
# save run callbacks & output. Avoids risk of overwriting previous run outputs
def hourstamp():
    return dt.now().strftime('%Y-%m-%d_%H')


# Sets global variables for image size, final layer size and the label_number
# of model layers to freeze for fine fine_tuning

IM_HEIGHT, IM_WIDTH = 299, 299  # was fixed size (299,299) (though NOT in Keras v2) for InceptionV3
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

##################################################
# 'CORE' FUNCTION CODE
##################################################

# 'SUPPORTING' FUNCTIONS for training

def get_nb_files(directory):
    '''
    Get number of files by searching directory recursively
    '''
    if not os.path.exists(directory):
            return 0
    filecount = 0
    for dirpath, dirs, files in os.walk(directory):
        for dr in dirs:
            # counts all files in dirs which match 'a/path/*' pattern
            filecount += len(glob.glob(os.path.join(dirpath, dr + "/*")))
    return filecount


def setup_to_transfer_learn(model, base_model):
    '''
    Freeze all layers and compile the model
    '''
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                    metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    '''
    Add new last layer to the CNN
    args:
    base_model: keras model excluding top layer
    nb_classes: # of classes

    Returns:
    new keras model with last layer
    '''
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    '''
    Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception
            blocks in the inceptionv3 arch

    Args:
    model: keras model
    '''
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                    loss='categorical_crossentropy', metrics=['accuracy'])


# The 'main' training function

def train(source_dir, train_dir='training_dir', val_dir='validation_dir',
          nb_epoch=3, batch_size=32,
          output_dir='./model_output_run_X_',
          output_model_file='Incv3_nested_model',
          model_to_load=None, chk_save_period=2,
          trf_learning=True, fine_tuning=False
          ):
    '''
    Use transfer learning and fine-tuning to train a network on a new dataset

    args:
    source_dir - str - path to data directory (containing training and validation sets)

    kwargs:
    train_dir - str - default 'training_dir' - dir name in source_dir for training dataset
    val_dir - str - default 'validation_dir' - dir name in source_dir for validation dataset
    nb_epoch - int - default=3 - number of epochs for training
    batch_size - int - default=32 - batch size for training
    output_dir - str - detault='./model_output_run_X_' - directory name for storing training outputs
    output_model_file - str - default='Incv3_nested_model' - name to save model, and stem
            for checkpoint save points
    model_to_load - str - default=None - path to existing model to load for training
        if None (default), will use InceptionV3
    chk_save_period - int - default=2 - how often (epochs) to save model checkpoints
    trf_learning - bool - default=True - run transfer learning
    fine_tuning - bool - default=False - run fine tuning

    '''
    # Define paths to training and validation directories
    train_dir = os.path.join(source_dir, train_dir)   # define full path to train_dir
    val_dir = os.path.join(source_dir, val_dir)   # define full path to val_dir

    # creates & sets output directory for all output files, with hourstamp
    # to avoid accidental overwriting from later runs
    output_dir = output_dir+hourstamp()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # count number of files and classes in training and validation data
    nb_train_samples = get_nb_files(train_dir)   # get number of files in train_dir
    nb_classes = len(glob.glob(train_dir + "/*"))   # get number of classes in train_dir
    nb_val_samples = get_nb_files(val_dir)   # get number of files in val_dir

    # set steps per epoch and validation steps based on # samples / batch_size
    steps_per_epoch = nb_train_samples / batch_size
    validation_steps = nb_val_samples / batch_size

    # data preparation
    # create ImageDataGenerators for train and validation data
    # run InceptionV3 preprocess_input to preprocess numpy array of images
    # then add range of image transforms (rotation, shift, shear, zoom, Hflip)
    print('Running data prep - train')
    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

    print('Running data prep - test')
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

    # sets up generator to create flow from directory for training_dir
    # to defined target_size and batch_size
    print('Running data gen - train')
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=batch_size
        )

    # extract class labels from train_generator as csv file
    print('Saving class labels')
    label_map = pd.DataFrame.from_dict(train_generator.class_indices,
                                        orient='index')
    label_map.rename(columns={0:'label_number'}, inplace=True)
    label_map_path = os.path.join(output_dir, 'class_label_map.csv')
    label_map.to_csv(label_map_path)

    # sets up generator to create flow from directory for validation_dir
    # to defined target_size and batch_size
    print('Running data gen - test')
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=batch_size
        )

    # SETUP Model
    # Load InceptionV3 (usual) or existing model (model_to_load)
    print('Setup model')
    # if no 'model_to_load' specified, then load InceptionV3
    # include_top=False allows new last layer for correct #classes
    # call add_new_last_layer to create correct top layer
    if not model_to_load:
        print('Starting with InceptionV3, imagenet weights')
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        model = add_new_last_layer(base_model, nb_classes)

    # if model_to_load specifed, then load that model
    # commonly used when selecting a model from a transfer learning run
    # to use as basis for fine tuning
    # NOTE only works for training on same dataset as model_to_load
    # as no layer changes / reconfig applied
    if model_to_load:
        print('Loading existing model: {}'.format(model_to_load))
        model = load_model(model_to_load)




    # SETUP CALLBACKS
    # Create directory for model checkpoints
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # checkpoint 'base' path is same as output_model_file
    checkpoint_save_path = os.path.join(checkpoint_dir, output_model_file)

    # checkpoint callbacks
    # filename = 'base' name + tl / ft + epoch + val_loss
    # default is to save every 2 epochs (chk_save_period)
    checkpoint_tl = ModelCheckpoint(checkpoint_save_path+'_tl_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val-loss', verbose=0,
                                save_best_only=False, period=chk_save_period)

    checkpoint_ft = ModelCheckpoint(checkpoint_save_path+'_ft_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val-loss', verbose=0,
                                save_best_only=False, period=chk_save_period)


    # history logging callbacks
    # streams epoch results to a csv file
    log_save_path = os.path.join(output_dir, 'history_log')
    csv_logger_tl = CSVLogger(log_save_path+'_tl.csv')
    csv_logger_ft = CSVLogger(log_save_path+'_ft.csv')


    # setup and run transfer learning
    if trf_learning:
        # transfer learning
        print('setup transfer learning')
        setup_to_transfer_learn(model, base_model)


        print('TRANSFER lEARN fit_generator')
        history_tl = model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=None,
            callbacks=[checkpoint_tl, csv_logger_tl]
            )

        pd.DataFrame(history_tl.history).to_csv(
            os.path.join(output_dir, 'history_tl.csv'))

    # setup and run fine tuning
    if fine_tuning:
        # fine-tuning
        print('setup finetune')
        setup_to_finetune(model)

        print('FINETUNE model fit generator')
        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=None,
            callbacks=[checkpoint_ft, csv_logger_ft])

        pd.DataFrame(history_ft.history).to_csv(
            os.path.join(output_dir, 'history_ft.csv'))

    # save final model file
    model.save(os.path.join(output_dir, output_model_file))





if __name__=='__main__':
    train('./data/model_data',                          # model data dir
            output_model_file='Incv3_nested_run_X',     # ouput model file
            model_to_load=None,                         # model to load (None for base InceptionV3)
            chk_save_period=2,                          # how often to save checkpoints (epochs)
            nb_epoch=3,                                 # how mnay epochs to run
            batch_size=32,                              # batch size
            trf_learning=True,                          # transfer learning?
            fine_tuning=False                           # fine tuning?
            )

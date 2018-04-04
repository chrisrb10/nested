import os
import sys
import glob
import pandas as pd
from datetime import datetime as dt
#import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


def hourstamp():
    return dt.now().strftime('%Y-%m-%d_%H')


IM_HEIGHT, IM_WIDTH = 299, 299 # was fixed size (299,299) (though NOT in Keras v2) for InceptionV3
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
            return 0
    filecount = 0
    for dirpath, dirs, files in os.walk(directory):
        for dr in dirs:
            # counts all files in dirs which match 'a/path/*' pattern
            filecount += len(glob.glob(os.path.join(dirpath, dr + "/*")))
    return filecount


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """
    Add last layer to the convnet
    args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(source_dir, train_dir='training_dir', val_dir='validation_dir',
          nb_epoch=3, batch_size=32,
          output_model_file='Incv3_nested_model',
          trf_learning=True, fine_tuning=True
          ):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    train_dir = os.path.join(source_dir, train_dir)   # define full path to train_dir
    val_dir = os.path.join(source_dir, val_dir)   # define full path to val_dir

    output_dir = './model_output_'+hourstamp()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    nb_train_samples = get_nb_files(train_dir)   # get number of files in train_dir
    nb_classes = len(glob.glob(train_dir + "/*"))   # get number of classes in train_dir
    nb_val_samples = get_nb_files(val_dir)   # get number of files in val_dir

    # NOTE - lines below probably redundant, hangover from args.* unpacking
    #nb_epoch = int(nb_epoch)   # get number of epochs for training
    #batch_size = int(batch_size)   # get batch size

    steps_per_epoch = nb_train_samples / batch_size
    validation_steps = nb_val_samples / batch_size

    # data prep
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

    print('Running data gen - train')
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=batch_size
        )


    # extract class labels as csv file
    print('Saving class labels')
    label_map = pd.DataFrame.from_dict(train_generator.class_indices,
                                        orient='index')
    label_map.rename(columns={0:'label_number'}, inplace=True)
    label_map.to_csv('class_label_map.csv')

    print('Running data gen - test')
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_HEIGHT, IM_WIDTH),
        batch_size=batch_size
        )

    # setup model
    print('Setup model')
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
                    #include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    # setup callbacks
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpoint_save_path = os.path.join(checkpoint_dir, output_model_file)

    checkpoint_tl = ModelCheckpoint(checkpoint_save_path+'_tl_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val-loss', verbose=0,
                                save_best_only=False, period=2)

    checkpoint_ft = ModelCheckpoint(checkpoint_save_path+'_ft_{epoch:02d}-{val_loss:.2f}.hdf5',
                                monitor='val-loss', verbose=0,
                                save_best_only=False, period=2)

    if trf_learning:
        # transfer learning
        print('setup transfer learning')
        setup_to_transfer_learn(model, base_model)


        print('trf learn fit_generator')
        history_tl = model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=None,       # change class_weight to None (from 'auto')
            callbacks=[checkpoint_tl]
            )

        pd.DataFrame(history_tl.history).to_csv(
            os.path.join(output_dir, 'history_tl.csv'))

    if fine_tuning:
        # fine-tuning
        print('setup finetune')
        setup_to_finetune(model)

        print('ft model fit generator')
        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            class_weight=None,       # change class_weight to None (from 'auto')
            callbacks=[checkpoint_ft])

        pd.DataFrame(history_ft.history).to_csv(
            os.path.join(output_dir, 'history_ft.csv'))

    model.save(os.path.join(output_dir, output_model_file))





if __name__=='__main__':
    train('./data/model_data/',
            output_model_file='Incv3_nested_run_1',
            nb_epoch=200,
            batch_size=32,
            trf_learning=False,
            fine_tuning=False)

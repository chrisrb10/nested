# nested
Room image classification with InceptionV3

This repo contains project material for work with online estate agent nested.com on a project to build a CNN classifier of interior room images from online property listings.

## `.py` scripts

**`nested_utilities.py`** 
Contains a number of utility functions relating to:
- Moving and manipulating large image datasets, and creating and configuring related directories
- Cataloguing (as dataframes)  directories of images grouped into sub-directories as image label, and using file_path to populate image labels
- Splitting datasets into test & train data, and (within train) into train and validate, and placing in directory structures ready for use by the CNN training script (`nested_inception.py`)

**`nested_inception.py`**
The main CNN training script.

**`nested_predict.py`**
Contains a number of functions to make predictions using the trained model, and extract labels and probabilities from resulting data

**`nested_plot.py`**
Contains a number of functions for plotting images & data relevant to the project.
A key function is `image_grid` which uses image_paths in a df to construct grids from a range of images meeting different parameters, and applies a range of labelling functions on them (with data from same dataframe / catalogue)


## Jupyter notebooks


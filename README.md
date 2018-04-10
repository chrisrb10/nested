# nested
Room image classification with InceptionV3

This repo contains project material for work with online estate agent nested.com on a project to build a CNN classifier of interior room images from online property listings.

## `.py` scripts

**`nested_utilities.py`**   
Contains a number of utility functions relating to:
- Moving and manipulating large image datasets, and creating and configuring related directories
- Cataloguing (as dataframes)  directories of images grouped into sub-directories as image label, and using file_path to populate image labels
- Splitting datasets into test & model data, and (within model) into train and validate, and placing in directory structures ready for use by the CNN training script (`nested_inception.py`)

**`nested_inception.py`**  
The main CNN training script.

**`nested_predict.py`**  
Contains a number of functions to make predictions using the trained model, and extract labels and probabilities from resulting data

**`nested_plot.py`**  
Contains a number of functions for plotting images & data relevant to the project.
A key function is `image_grid` which uses image_paths in a df to construct grids from a range of images meeting different parameters, and applies a range of labelling functions on them (with data from same dataframe / catalogue)


## Jupyter notebooks

**`processing_master_1_sort_catalogue.ipynb`**  
Documents the work to prepare the raw library of labelled images for modelling. 
Mostly focussed on cataloguing the library, reviewing & reclassifying labelling errors, and excluding a small number of images

**`processing_master_2_dir_split.ipynb`**  
Documents the splitting of data into test / model, and then the model data into train/validate

**`reprocess_test_data.ipynb`**   
A second review & reclassification workbook for the test data, fix mislabelling errors not captured in first processing, and highlighted by prediction & evaluation.

**`prediction.ipynb`**  
Documents using the trained model to predict for the held out test data

**`evaluation.ipynb`**  
Evaluates the model labelling of the test data

## Other directories / files

**`/data_catalogues/`**  
Contained various catalogues (pandas dataframes saved as `.csv`) from processing, relassifying and test_data predictions.

**`/image_grids/`**   
contains a range of different composite image grids prepared during data processing and prediction evalation   

**`/plots`**    
Contains plots related to analysis & evaluation   

**`/presentations/`**   
Contains presentations and presentation material relating to the project



from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'sex', # M, F, and I (infant)
    'length', # Longest shell measurement
    'diameter', # perpendicular to length
    'height', # with meat in shell
    'whole_weight', # whole abalone
    'shucked_weight', # weight of meat
    'viscera_weight', # gut weight (after bleeding)
    'shell_weight'] # after being dried

label_column = 'rings'

feature_columns_dtype = {
    'sex': "category",
    'length': "float64",
    'diameter': "float64",
    'height': "float64",
    'whole_weight': "float64",
    'shucked_weight': "float64",
    'viscera_weight': "float64",
    'shell_weight': "float64"}

label_column_dtype = {'rings': "float64"} # +1.5 gives the age in years

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

if __name__ == '__main__':
    
    """
    1. read the raw data
    2. pre-process it i.e fit sklearn preprocess pipeline on it
    3. PERSIST THE TRAINED preprocess MODEL
    4. exit
    
    In nutshell just get a trained sklearn-pre-processor
    """

    # step_1 : parse the command line inputs
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    
    # step_2 : load raw df from args.train path
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)) for file in input_files ]
    concat_data = pd.concat(raw_data)

    # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
    concat_data.drop(label_column, axis=1, inplace=True)
    print("{}_training_the_transformModel_{}".format("="*40, "="*40))
    print("data shape : ", concat_data.shape)
    
    # step_3 : pre-processor
    # This section is adapted from the scikit-learn example of using preprocessing pipelines:
    #
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    #
    # We will train our classifier with the following features:
    # Numeric Features:
    # - length:  Longest shell measurement
    # - diameter: Diameter perpendicular to length
    # - height:  Height with meat in shell
    # - whole_weight: Weight of whole abalone
    # - shucked_weight: Weight of meat
    # - viscera_weight: Gut weight (after bleeding)
    # - shell_weight: Weight after being dried
    # Categorical Features:
    # - sex: categories encoded as strings {'M', 'F', 'I'} where 'I' is Infant
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore'))

    preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="category"))])
    
    # step_4 : fit the pre-processor
    print("imp : shape of data before pp: ", concat_data.shape)
    preprocessor.fit(concat_data)
    
    features = preprocessor.transform(concat_data)
    print("imp : shape of data after pp: ", features.shape)
    print("sample data : \n", features[0])
    
    # step_5 : persist the model
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, content_type):
    """
    std func : provide input df to the pre-processor <*ONLY during transform/predict>
        i.e load data from S3, add column names to it
       
    if col_to_predict present in input_data:
        appned it to and pass on:
    else:
        process without col_to_predict
        
    Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    print("{} input_fn {}".format("="*40, "="*40))
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), 
                         header=None)
        
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            print("This is a labelled example, includes the col_to_predict")
            print("df.shape : ", df.shape)
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            print("This is an unlabelled example.")
            print("df.shape : ", df.shape)
            df.columns = feature_columns_names
            
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    print("{} output_fn {}".format("="*40, "="*40))
    
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """
    imppp : input_data : comming from input_fn
    
    std func to execute preprocessor.transform(), fit() funcs.
    
    Preprocess input data
    
    if col_to_predict present in input_data:
        appned it to and pass on:
    else:
        process without col_to_predict
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    print("{} predict_fn {}".format("="*40, "="*40))
    
    # it anyhow filter out the label_column columns (as per the script)
    print("imp : raw data shape : ".format(input_data.shape))
    features = model.transform(input_data)
    
    if label_column in input_data:
        # this section used for training (NOT pred/pipeline)
        features = np.insert(features, 0, input_data[label_column], axis=1)
        print("training job")
        print("imp : pp feature data shape : {}".format(features.shape))
        print("sample data : \n", features[0])
        # if it is traning job
        # Return the label (as the first column) and the set of features.
        return features
    else:
        # this section used for pred/pipeline (Imp), here pass the input data without col_to_predict
        print("test/pred job")
        print("imp : pp feature data shape : {}".format(features.shape))
        print("sample data : \n", features[0])
        # if it is test/pred job
        # Return only the set of features
        return features
    
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor

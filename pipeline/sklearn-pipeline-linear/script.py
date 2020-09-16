
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf



if __name__ =='__main__':

    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, default=10)
    parser.add_argument('--min-samples-leaf', type=int, default=3)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    #parser.add_argument('--train-file', type=str, default='boston_train.csv')
    #parser.add_argument('--test-file', type=str, default='boston_test.csv')
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str) # in this script we ask user to explicitly name the target

    args, _ = parser.parse_known_args()
    
    print("args.train", args.train)
    print("args.test", args.test)
    
    print('reading train data')
    print("args.train : ",args.train)
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_df = pd.concat(raw_data)
    print(train_df.shape)
    
    print('reading test data')
    print("args.test : ",args.test)
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.test, file) for file in os.listdir(args.test) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    test_df = pd.concat(raw_data)
    print(test_df.shape)

    print('building training and testing datasets')
    """
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target]
    y_test = test_df[args.target]
    """
    print(train_df.columns.values)
    col_to_predict = train_df.columns.values[0]
    print("col_to_predict : {}, arg_type : {}".format(col_to_predict, type(col_to_predict)))
    X_train = train_df.drop(columns=[col_to_predict])
    X_test = test_df.drop(columns=[col_to_predict])
    y_train = train_df[col_to_predict]
    y_test = test_df[col_to_predict]
    
    
    # train
    print('training model')
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1)
    
    print("-"*100)
    print("X_train.shape : ", X_train.shape)
    print("model training on num features : ", X_train.shape[1])
    print("sample data : \n", X_train.head(1).values)
    model.fit(X_train, y_train)

    # print abs error
    print('validating model')
    abs_err = np.abs(model.predict(X_test) - y_test)

    # print couple perf metrics
    for q in [10, 50, 90]:
        print('AE-at-' + str(q) + 'th-percentile: '
              + str(np.percentile(a=abs_err, q=q)))
        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model persisted at ' + path)
    print(args.min_samples_leaf)

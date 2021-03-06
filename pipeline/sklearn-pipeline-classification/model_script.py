import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-q", "-m", "pip", "install", package])
    
install('matplotlib')
install('seaborn')


import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def classification_metrices(y_true=None, y_pred=None, target_names=None, verbose=True):
    cr = classification_report(y_true, y_pred, target_names=target_names)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall    = recall_score(y_true, y_pred, average='macro')

    print("accuracy  : {}".format(round(accuracy,2)))
    print("precision : {}".format(round(precision,2)))
    print("recall    : {}".format(round(recall,2)))
    print("\nClassification report : \n{}".format(cr))
    
    return accuracy, precision, recall, cr



def plot_cm(y_pred=None, y_true=None, path_saveplot=None, show_plot=True, plot_size=(8,6) ):
    cf_matrix = confusion_matrix(y_true, y_pred)

    ac = accuracy_score( y_true, y_pred )
    all_vals = precision_recall_fscore_support(y_true, y_pred )
    precision = all_vals[0][1]
    recall = all_vals[1][1]
    fscore = all_vals[2][1]
    support = all_vals[3][1]

    text_print_plot = \
    """
    Confusion Matrix
    {} = {} 
    {} = {}, {} = {}
    {} = {}
    {} = {}
    """.format(
        'Accuracy', round(ac,2), 
        'Precision', round(precision,2), 
        'Recall', round(recall, 2),
        'Fscore', round(fscore, 2),
        'Support', support
    )
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.1%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n\n{v2}\n\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure( figsize=plot_size )
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    
    plt.title(text_print_plot, fontweight='bold', fontsize=14)
    plt.xlabel('pred', fontsize=14)
    plt.ylabel('true', fontsize=14)
    plt.tight_layout()
    
    path = os.path.join(path_saveplot, "model_cm_plot.png")
    plt.savefig( path )
    if( show_plot ):
        plt.show()
    
    plt.close()
    
    
    
def plot_pr_curve(y_score=None, y_true=None, path_saveplot=None, show_plot=True, plot_size=(6,5)):

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision    = average_precision_score(y_true, y_score)
    
    
    plt.figure( figsize=plot_size )
    sns.set_style('darkgrid')
    plt.step(recall, precision, 'black', color='black', alpha=0.7, where='post', label='AUC = %0.2f' % average_precision )
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.legend(loc = 'lower right')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision), fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    path = os.path.join(path_saveplot, "model_pr_curve_plot.png")
    plt.savefig( path )
    if( show_plot ):
        plt.show()
    
    plt.close()
        
    return precision, recall, plt


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
    print(train_df.head(1))
    
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
    print("columns : ", train_df.columns.values)
    
    col_to_predict = train_df.columns.values[0]
    print("col_to_predict : {}, arg_type : {}".format(col_to_predict, type(col_to_predict)))
    X_train = train_df.drop(columns=[col_to_predict])
    X_test = test_df.drop(columns=[col_to_predict])
    y_train = train_df[col_to_predict]
    y_test = test_df[col_to_predict]
    
    print(type(y_train))
    arr = y_train.value_counts()
    print("train : {} : {}".format(arr.index.values, arr.values))
    arr = y_test.value_counts()
    print("test  : {} : {}".format(arr.index.values, arr.values))
    
    # train
    print('training model')
    model = RandomForestClassifier(
                                    n_estimators=args.n_estimators,
                                    min_samples_leaf=args.min_samples_leaf,
                                    n_jobs=-1
                                    )
    
    print("-"*100)
    print("X_train.shape : ", X_train.shape)
    print("model training on num features : ", X_train.shape[1])
    print("sample data : \n", X_train.head(1).values)
    model.fit(X_train, y_train)
    
    y_true = y_test
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    target_names = ["small", "large"]

    # print classification report
    accuracy, precision, recall, cr = classification_metrices(y_true=y_true, y_pred=y_pred, 
                                                          target_names=target_names, verbose=True)
    
    
    # print auc_pr curve
    plot_cm(y_pred=y_pred, y_true=y_true, path_saveplot=args.model_dir, show_plot=False, plot_size=(8,6) )
    plot_pr_curve(y_true=y_true, y_score=y_score, path_saveplot=args.model_dir, show_plot=False, plot_size=(8,6))
    
        
    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print('model persisted at ' + path)
    print(args.min_samples_leaf)
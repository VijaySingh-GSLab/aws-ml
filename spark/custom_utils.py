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


CUSTOM_STOP_WORDS = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                  'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amount', 'an',
                  'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren',
                  "aren't", 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
                  'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond',
                  'both', 'bottom', 'but', 'by', 'ca', 'call', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd',
                  'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'done', 'down',
                  'due', 'during', 'each', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough',
                  'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
                  'fifty', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'four', 'from', 'front', 'full',
                  'further', 'get', 'give', 'go', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven',
                  "haven't", 'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon',
                  'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed',
                  'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'keep', 'last', 'latter',
                  'latterly', 'least', 'less', 'll', 'm', 'ma', 'made', 'make', 'many', 'may', 'me', 'meanwhile',
                  'might', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must',
                  'mustn', "mustn't", 'my', 'myself', 'name', 'namely', 'needn', "needn't", 'neither', 'never',
                  'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
                  'nowhere', 'o', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others',
                  'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please',
                  'put', 'quite', 'rather', 're', 'really', 'regarding', 's', 'same', 'say', 'see', 'seem', 'seemed',
                  'seeming', 'seems', 'serious', 'several', 'shan', "shan't", 'she', "she's", 'should', "should've",
                  'shouldn', "shouldn't", 'show', 'side', 'since', 'six', 'sixty', 'so', 'some', 'somehow', 'someone',
                  'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 't', 'take', 'ten', 'than',
                  'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
                  'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third', 'this',
                  'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top',
                  'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'unless', 'until', 'up', 'upon', 'us',
                  'used', 'using', 'various', 've', 'very', 'via', 'was', 'wasn', "wasn't", 'we', 'well', 'were',
                  'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
                  'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever',
                  'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn',
                  "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
                  'yourselves']



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
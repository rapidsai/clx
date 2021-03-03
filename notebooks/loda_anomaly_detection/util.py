import numpy as np 
from cuml.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import roc_curve
import cupy as cp
import matplotlib.pylab as plt 

def average_precision_score(y_true, y_score):
    """
    Compute average precision score using precision and recall computed from cuml. 
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score) 
    # return step function integral 
    return -cp.sum(cp.diff(recall) * cp.array(precision)[:-1])

def metrics(y_true, y_score):
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    ap = average_precision_score(y_true, y_score)
    return [auc, ap]

def plot_roc(label, y_scores):
    fpr, tpr, _ = roc_curve(y_true=label.values.tolist(), y_score=y_scores.tolist())    
    auc = metrics(label, y_scores)[0]
    plt.plot(fpr, tpr, label="ROC = " + str(np.round(auc,2)))
    plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'r-')
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.legend(loc='best')
    plt.title('Area under AUC curve')

def plot_pr(label, y_scores):
    ap = metrics(label, y_scores)[1]
    precision, recall, _ = precision_recall_curve( label, y_scores)
    plt.plot(recall, precision, label='AP = ' + str(np.round(ap,2)))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='best')
    plt.title('Area under PR curve')
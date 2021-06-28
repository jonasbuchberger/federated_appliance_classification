import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def metrics_table(y_true, y_pred, class_dict):
    """ Creates dataframe with all metrics for each appliance and also the mean

    Args:
        y_true (list): List of true labels
        y_pred (list): List of predicted labels
        class_dict (dict):

    Returns:
        (pd.Dataframe): Dataframe with metrics
    """

    # Remove classes with no samples
    class_type = {v: k for k, v in class_dict.items()}
    tmp_dict = {}
    for label in set(y_true):
        tmp_dict[class_type[label]] = label
    class_dict = tmp_dict

    class_dict_values = list(class_dict.values())
    class_dict_keys = list(class_dict.keys())

    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(y_true, y_pred,
                                                                                             labels=class_dict_values)
    matrix = confusion_matrix(y_true, y_pred, labels=class_dict_values)
    class_acc = matrix.diagonal() / matrix.sum(axis=1)
    overall_accuracy = accuracy_score(y_true, y_pred)

    df = pd.DataFrame({'Accuracy': np.append(class_acc, class_acc.mean()),
                       'F1': np.append(class_f1, class_f1.mean()),
                       'Precision': np.append(class_precision, class_precision.mean()),
                       'Recall': np.append(class_precision, class_recall.mean()),
                       'Overall Accuracy': overall_accuracy},
                      index=class_dict_keys + ['Mean'])

    return df


if __name__ == '__main__':
    y_true = [1,1,1,3,3,3,4,4,4]
    y_pred = [1,1,1,3,3,3,4,4,4]
    class_dict = {
        'a': 1,
        'b': 2,
        'c': 3,
        'e': 4
    }
    print(metrics_table(y_true, y_pred, class_dict))
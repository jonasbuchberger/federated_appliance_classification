import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def metrics_table(y_true, y_pred, class_dict):
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

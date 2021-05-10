import io

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torchvision import transforms as torch_transforms

def tensor_confusion_matrix(y_true, y_pred, class_dict):
    """ Creates confusion matrix for labels.

    Args:
        y_true (torch.Tensor): Binary targets
        y_pred (torch.Tensor): Predicted classes
        class_dict (dict): Dict of the class mapping
    Returns:
        (torch.Tensor): Image tensor of the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=torch.arange(len(class_dict.keys())))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_dict.keys()))
    disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='vertical', values_format='d')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf).copy()
    buf.close()

    to_tensor = torch_transforms.ToTensor()
    im = to_tensor(im)

    return im

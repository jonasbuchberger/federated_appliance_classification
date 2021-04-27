import os
from pathlib import Path

import torch
from tensorboard import program
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class SummaryWriter(SummaryWriter):
    """
        https://github.com/pytorch/pytorch/issues/32651
    """
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def start_tensorboard(log_dir, system='local'):
    """ Starts Tensorboard server

    Args:
        log_dir (str): Path to tensorboard logging folder
        system (str): Local or remote tensorboard server
    """
    tb = program.TensorBoard()
    if system == 'remote':
        tb.configure(argv=[None, '--logdir', log_dir, '--host', '0.0.0.0'])
    else:
        tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f'TensorBoard started: {url}')
    input("Press Enter to end TensorBoard.")


def start_jupyter():
    """ Starts remote Jupyter server

    """
    os.system('jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''')
    input("Press Enter to end Jupyter.")


ROOT_DIR = Path(__file__).parent.parent

if __name__ == '__main__':
    start_tensorboard(os.path.join(ROOT_DIR, 'models'))
    # start_jupyter()

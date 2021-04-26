import os
from pathlib import Path

from tensorboard import program


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

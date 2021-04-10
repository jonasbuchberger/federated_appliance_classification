from pathlib import Path
from tensorboard import program
import os

def start_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--host', '0.0.0.0'])
    url = tb.launch()
    print(f'TensorBoard started: {url}')
    input("Press Enter to end TensorBoard.")

def start_jupyter():
    os.system('jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''')

ROOT_DIR = Path(__file__).parent.parent

if __name__ == '__main__':
    start_tensorboard(os.path.join(ROOT_DIR, 'models'))
    #start_jupyter()

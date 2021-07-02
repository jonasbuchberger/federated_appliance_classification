import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from src.utils import ROOT_DIR

def pi_bytes(log_path):

    log_path = os.path.join(log_path, 'pi_logs')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.suptitle('Network traffic')
    ticks = None
    time_arr = None
    labels = []
    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        mb_sent = (df['bytes_sent'] - df['bytes_sent'][0]) * 1e-6
        axes[0].plot(mb_sent)
        labels.append(f'Pi{log[5:-4]}')
        if ticks is None:
            time = (datetime.fromisoformat(df['time'].iloc[-1]) - datetime.fromisoformat(df['time'].iloc[0]))
            time = int(time.total_seconds() / 60)
            ticks = np.arange(0, len(df), int(len(df)/5))
            time_arr = np.arange(0, time, time / len(ticks), dtype=int)

    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(time_arr)
    axes[0].set_ylabel('Sent data in Mbyte')

    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        mb_sent = (df['bytes_recv'] - df['bytes_recv'][0]) * 1e-6
        axes[1].plot(mb_sent)

    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(time_arr)
    axes[1].set_ylabel('Received data in Mbyte')

    fig.text(0.5, 0, 'Time in minutes', ha='center', va='center')

    lgd = fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=5)
    plt.tight_layout()
    plt.savefig('bytes', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def pi_cpu(log_path):
    log_path = os.path.join(log_path, 'pi_logs')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plt.suptitle('CPU information')
    ticks = None
    time_arr = None
    labels = []

    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        cpu_user = (df['cpu0_user'] + df['cpu1_user'] + df['cpu2_user'] + df['cpu3_user']) / 4
        axes[0].plot(cpu_user)
        labels.append(f'Pi{log[5:-4]}')
        if ticks is None:
            time = (datetime.fromisoformat(df['time'].iloc[-1]) - datetime.fromisoformat(df['time'].iloc[0]))
            time = int(time.total_seconds() / 60)
            ticks = np.arange(0, len(df), int(len(df) / 5))
            time_arr = np.arange(0, time, time / len(ticks), dtype=int)

    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(time_arr)
    axes[0].set_ylabel('CPU usage in %')

    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        cpu_freq = (df['cpu0_freq'] + df['cpu1_freq'] + df['cpu2_freq'] + df['cpu3_freq']) / 4 / 1000
        axes[1].plot(cpu_freq)

    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(time_arr)
    axes[1].set_ylabel('CPU frequency in GHz')

    fig.text(0.5, 0, 'Time in minutes', ha='center', va='center')

    lgd = fig.legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=5)
    plt.tight_layout()
    plt.savefig('cpu', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    log_dir = os.path.join(ROOT_DIR, "models/Federated_16_DENSE_SGD_CrossEntropyLoss_CLASS_10_[RandomAugment_MFCC]/lr-0.059_wd-0.001_nl-4_ss-22_te-40_le-1")
    pi_bytes(log_dir)
    pi_cpu(log_dir)

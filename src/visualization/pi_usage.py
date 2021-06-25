import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from src.utils import ROOT_DIR

def pi_bytes(log_path):

    log_path = os.path.join(log_path, 'pi_logs')
    plt.figure(figsize=(10, 4))
    ticks = None
    time_arr = None
    plt.subplot(1, 2, 1)
    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        mb_sent = (df['bytes_sent'] - df['bytes_sent'][0]) * 1e-6
        plt.suptitle('Network traffic')
        plt.plot(mb_sent, label=f'Pi{log[5:-4]}')
        if ticks is None:
            time = (datetime.fromisoformat(df['time'].iloc[-1]) - datetime.fromisoformat(df['time'].iloc[0]))
            time = int(time.total_seconds() / 60)
            ticks = np.arange(0, len(df), int(len(df)/5))
            time_arr = np.arange(0, time, time / len(ticks), dtype=int)
            plt.xticks(ticks=ticks, labels=time_arr)
    plt.ylabel('Sent data in Mbyte')
    plt.xlabel('Time in minutes')

    plt.subplot(1, 2, 2)
    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        mb_sent = (df['bytes_recv'] - df['bytes_recv'][0]) * 1e-6
        plt.plot(mb_sent)
        plt.xticks(ticks=ticks, labels=time_arr)
    plt.ylabel('Received data in Mbyte')
    plt.xlabel('Time in minutes')

    lgd = plt.figlegend(loc=(0.25, -0.1), ncol=5, labelspacing=0.)
    plt.tight_layout()
    #plt.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def pi_cpu(log_path):
    log_path = os.path.join(log_path, 'pi_logs')
    plt.figure(figsize=(10, 4))
    ticks = None
    time_arr = None
    plt.subplot(1, 2, 1)
    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        plt.suptitle('CPU information')
        cpu_user = (df['cpu0_user'] + df['cpu1_user'] + df['cpu2_user'] + df['cpu3_user']) / 4
        plt.plot(cpu_user, label=f'Pi{log[5:-4]}')
        if ticks is None:
            time = (datetime.fromisoformat(df['time'].iloc[-1]) - datetime.fromisoformat(df['time'].iloc[0]))
            time = int(time.total_seconds() / 60)
            ticks = np.arange(0, len(df), int(len(df) / 5))
            time_arr = np.arange(0, time, time / len(ticks), dtype=int)
            plt.xticks(ticks=ticks, labels=time_arr)
    plt.ylabel('CPU usage in %')
    plt.xlabel('Time in minutes')

    plt.subplot(1, 2, 2)
    for i, log in enumerate(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, log))
        cpu_freq = (df['cpu0_freq'] + df['cpu1_freq'] + df['cpu2_freq'] + df['cpu3_freq']) / 4 / 1000
        plt.plot(cpu_freq)
        plt.xticks(ticks=ticks, labels=time_arr)
    plt.ylabel('CPU frequency in GHz')
    plt.xlabel('Time in minutes')

    lgd = plt.figlegend(loc=(0.25, -0.1), ncol=5, labelspacing=0.)
    plt.tight_layout()
    # plt.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    log_dir = os.path.join(ROOT_DIR, "models/Federated_16_CNN1D_SGD_CrossEntropyLoss_CLASS_10_[RandomAugment_MFCC]/lr-0.001_wd-0.0_nl-4_ss-28_te-60_le-5")
    pi_bytes(log_dir)
    pi_cpu(log_dir)

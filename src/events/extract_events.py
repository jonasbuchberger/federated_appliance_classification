import os
import pandas as pd
import datetime
import h5py
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock
from functools import partial


def calibrate_offset(data, average_frequency):
    """ Calibrates the offset to subtract of each measurement.
        Taken: https://zenodo.org/record/838974#.YMsl-mgzYuU

    Args:
        data (h5py.File): Dataset of the measurements
        average_frequency (int): Measurement frequency

    Returns:
        (np.array): Offset to apply on voltage
        (np.array): Offset to apply on current
    """
    if 'voltage' not in list(data):
        return np.zeros(data['voltage1'].shape), np.zeros(data['voltage1'].shape)

    length = len(data['voltage'])
    period_length = round(average_frequency / 50)

    remainder = divmod(length, period_length)[1]
    if remainder == 0:
        remainder = period_length

    offset = np.pad(data['voltage'][:],
                    (0, period_length - remainder),
                    'constant',
                    constant_values=0).reshape(-1, period_length).mean(axis=1)

    x = np.linspace(1, length, length // period_length, dtype=int)
    new_x = np.linspace(1, length, length - period_length, dtype=int)
    offset = scipy.interpolate.interp1d(x, offset)(new_x)
    offset = np.concatenate(
        (np.repeat([offset[0]], period_length / 2), offset, np.repeat([offset[-1]], period_length / 2)))
    return offset, offset * 0.7


def process_signal(data, socket_id):
    """ Applies the offset correction and mean subtraction.
        Also applies a median filter and the calibration.
        As suggested in the BLOND paper.

    Args:
        data (h5py.File): Dataset of the measurements
        socket_id (int): The socket to pe processed

    Returns:
        (np.array): The processed current
        (np.array): The processed voltage
    """
    current = data[f"current{socket_id}"]
    voltage = data['voltage']
    offset_voltage, offset_current = calibrate_offset(data, data.attrs['frequency'])
    current -= offset_current
    current -= np.mean(current)
    current = scipy.signal.medfilt(current, 15)
    voltage -= offset_voltage
    voltage -= np.mean(voltage)
    voltage = scipy.signal.medfilt(voltage, 15)

    current = current * data[f"current{socket_id}"].attrs['calibration_factor']
    voltage = voltage * data['voltage'].attrs['calibration_factor']

    return current, voltage


def process_event(path_to_data, dest_path, label, measurement_frequency=6400, snippet_length=25600, verbose=False):
    """ Cuts the event snippets out of the data.
        Applies preprocessing.

    Args:
        path_to_data (string): Path to BLOND dataset
        dest_path (string): Destination of the snippets
        label (int, pd.Series): Row of the events dataframe
        measurement_frequency (int): Frequency of the measurements
        snippet_length (int): Length in measurements of the extracted snippets
        verbose (bool): Stores a plot of each event
    """
    _, label = label

    medal_id = label['Medal_Nr']
    date = str(label['Date'])
    event_timestamp = datetime.datetime.strptime(label['Timestamp'][:-3] + '00', '%Y-%m-%d %H:%M:%S.%f%z')
    medal_path = os.path.join(path_to_data, date, f'medal-{medal_id}')

    files = sorted(os.listdir(medal_path))
    for i, file in enumerate(files):
        file_timestamp = file.split(f'medal-{medal_id}-')[1][:-13]
        file_timestamp = datetime.datetime.strptime(file_timestamp, '%Y-%m-%dT%H-%M-%S.%fT%z')

        if file_timestamp > event_timestamp:
            file_timestamp = files[i - 1].split(f'medal-{medal_id}-')[1][:-13]
            file_timestamp = datetime.datetime.strptime(file_timestamp, '%Y-%m-%dT%H-%M-%S.%fT%z')

            appliance = label['Appliance_Name']
            medal_id = label['Medal_Nr']
            socket_id = label['Socket_Nr']
            class_name = label['Class_Name']
            timestamp = label['Timestamp'].replace(':', '_')

            event_file = f'{medal_id}_{socket_id}_{appliance}_{timestamp}.h5'
            event_file_path = os.path.join(dest_path, f'event_snippets/medal-{medal_id}', event_file)
            os.makedirs(os.path.join(dest_path, f'event_snippets/medal-{medal_id}'), exist_ok=True)

            # Check if file is already existing
            if not os.path.isfile(event_file_path):

                # Load file from FTP
                file_path = os.path.join(medal_path, files[i - 1])
                data = h5py.File(file_path, 'r')
                current, voltage = process_signal(data, socket_id)
                data.close()

                # Put event timestamp in the middle of the created event window
                event_index = int((event_timestamp - file_timestamp).total_seconds() * measurement_frequency)
                window_start_index = int(event_index - snippet_length / 2)
                window_end_index = int(event_index + snippet_length / 2)

                # Add file before or after event to extend to valid window size
                if window_start_index < 0:
                    print(f'Added before: {event_file}')
                    current = current[0:window_end_index]
                    voltage = voltage[0:window_end_index]

                    # Load file from FTP
                    file_path = os.path.join(medal_path, files[i - 2])
                    data = h5py.File(file_path, 'r')
                    current_2, voltage_2 = process_signal(data, socket_id)
                    data.close()
                    current = np.append(current_2[window_start_index:], current)
                    voltage = np.append(voltage_2[window_start_index:], voltage)

                elif window_end_index > len(current):
                    print(f'Added behind: {event_file}')
                    current = current[window_start_index:]
                    voltage = voltage[window_start_index:]

                    # Load file from FTP
                    file_path = os.path.join(medal_path, files[i])
                    data = h5py.File(file_path, 'r')
                    current_2, voltage_2 = process_signal(data, socket_id)
                    data.close()
                    current = np.append(current, current_2[:snippet_length - len(current)])
                    voltage = np.append(voltage, voltage_2[:snippet_length - len(voltage)])

                else:
                    current = current[window_start_index:window_end_index]
                    voltage = voltage[window_start_index:window_end_index]

                assert len(current) == snippet_length

                if verbose:
                    fig_path = os.path.join(dest_path, 'event_snippets', 'figs')
                    os.makedirs(fig_path, exist_ok=True)
                    lock.acquire()
                    plt.plot(current)
                    plt.savefig(os.path.join(fig_path, event_file.split('.')[0]))
                    plt.close()
                    lock.release()

                # Append new row to dataframe
                f = h5py.File(event_file_path, "w")
                f.create_dataset('data/block0_values', data=(np.stack((voltage, current), axis=1)))
                f.close()

                # Extend existing csv or create new one
                new_row = {'Medal': medal_id, 'Socket': socket_id, 'Appliance': appliance,
                           'Type': class_name, 'Timestamp': timestamp}
                csv_path = os.path.join(dest_path, 'event_snippets/events_new.csv')

                lock.acquire()
                if os.path.isfile(csv_path):
                    df_old = pd.read_csv(csv_path, index_col=0)
                    df_old = df_old.append(new_row, ignore_index=True)
                    df_old.to_csv(csv_path)
                else:
                    df = pd.DataFrame(columns=['Medal', 'Socket', 'Appliance', 'Type', 'Timestamp'])
                    df = df.append(new_row, ignore_index=True)
                    df.to_csv(csv_path)
                lock.release()
            break


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    # path_to_data = "C:/Users/jonas/Documents/MATLAB/BLOND"
    path_to_data = "/mnt/nilm/nilm/i13-dataset/BLOND/BLOND-50"
    dest_path = "/mnt/nilm/temp/buchberger/"

    labels = pd.read_csv('/home/ubuntu/federated_blond/data/csv/medal-13.csv')

    l = Lock()
    pool = Pool(processes=16, initializer=init, initargs=(l,))
    wrapper = partial(process_event, path_to_data, dest_path, verbose=True)
    result = pool.map_async(wrapper, labels.iterrows())
    result.get()

import os
import pandas as pd
import datetime
import h5py
import numpy as np
from ftplib import FTP
import scipy
import scipy.signal
from tqdm import tqdm
from multiprocessing import Pool, Lock
from functools import partial
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


def calibrate_offset(f, average_frequency):
    if 'voltage' not in list(f):
        return np.zeros(f['voltage1'].shape), np.zeros(f['voltage1'].shape)

    length = len(f['voltage'])
    period_length = round(average_frequency / 50)

    remainder = divmod(length, period_length)[1]
    if remainder == 0:
        remainder = period_length

    offset = np.pad(f['voltage'][:],
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


def process_event(label):
    measurement_frequency = 6400
    snippet_length = 25600

    _, label = label
    ip = '138.246.224.34'
    user = 'm1375836'
    passwd = 'm1375836'

    socket_base = 'BLOND/BLOND-50/'
    medal_id = label['Medal_Nr']
    date = str(label['Date'])
    event_timestamp = datetime.datetime.strptime(label['Timestamp'][:-6], '%Y-%m-%d %H:%M:%S.%f')

    ftp_socket = FTP(ip)
    ftp_socket.login(user, passwd)

    medal_path = os.path.join(socket_base, date + f'/medal-{medal_id}')
    ftp_socket.cwd(medal_path)

    files = sorted(ftp_socket.nlst())
    # Iterate over all files on FTP to find file with timestamp of event
    for i, file in enumerate(files):
        file_timestamp = '_'.join(file.split('-')[2:-1])
        file_timestamp = datetime.datetime.strptime(file_timestamp[:-6], '%Y_%m_%dT%H_%M_%S.%f')

        if file_timestamp > event_timestamp:
            file_timestamp = '_'.join(files[i - 1].split('-')[2:-1])
            file_timestamp = datetime.datetime.strptime(file_timestamp[:-6], '%Y_%m_%dT%H_%M_%S.%f')

            appliance = label['Appliance_Name']
            medal_id = label['Medal_Nr']
            socket_id = label['Socket_Nr']
            class_name = label['Class_Name']
            timestamp = label['Timestamp'].replace(':', '_')

            event_file = f'{medal_id}_{socket_id}_{appliance}_{timestamp}.h5'
            event_file_path = os.path.join(ROOT_DIR, f'data/event_snippets_new/medal-{medal_id}', event_file)
            os.makedirs(os.path.join(ROOT_DIR, f'data/event_snippets_new/medal-{medal_id}'), exist_ok=True)

            # Check if file is already existing
            if not os.path.isfile(event_file_path):

                # Load file from FTP
                tmp_path = os.path.join(ROOT_DIR, 'data/tmp', medal_path)
                os.makedirs(tmp_path, exist_ok=True)
                file_path = os.path.join(tmp_path, files[i - 1])
                data = open(file_path, 'wb')
                ftp_socket.retrbinary('RETR %s' % files[i - 1], data.write)
                data.close()

                data = h5py.File(os.path.join(file_path), 'r')
                current, voltage = process_signal(data, socket_id)
                data.close()

                # Put event timestamp in the middle of the created event window
                event_index = int((event_timestamp - file_timestamp).total_seconds() * measurement_frequency)
                window_start_index = int(event_index - snippet_length / 2)
                window_end_index = int(event_index + snippet_length / 2)

                # Add file before or after event to extend to valid window size
                if window_start_index < 0:
                    print(f'Added before: {event_file_path}')
                    current = current[0:window_end_index]
                    voltage = voltage[0:window_end_index]

                    # Load file from FTP
                    file_path = os.path.join(tmp_path, files[i - 2])
                    data = open(file_path, 'wb')
                    ftp_socket.retrbinary('RETR %s' % files[i - 2], data.write)
                    data.close()

                    data = h5py.File(os.path.join(file_path), 'r')
                    current_2, voltage_2 = process_signal(data, socket_id)
                    data.close()
                    current = np.append(current_2[window_start_index:], current)
                    voltage = np.append(voltage_2[window_start_index:], voltage)

                elif window_end_index > len(current):
                    print(f'Added behind: {event_file_path}')
                    current = current[window_start_index:]
                    voltage = voltage[window_start_index:]

                    # Load file from FTP
                    file_path = os.path.join(tmp_path, files[i])
                    data = open(file_path, 'wb')
                    ftp_socket.retrbinary('RETR %s' % files[i], data.write)
                    data.close()

                    data = h5py.File(os.path.join(file_path), 'r')
                    current_2, voltage_2 = process_signal(data, socket_id)
                    data.close()
                    current = np.append(current, current_2[:snippet_length - len(current)])
                    voltage = np.append(voltage, voltage_2[:snippet_length - len(voltage)])

                else:
                    current = current[window_start_index:window_end_index]
                    voltage = voltage[window_start_index:window_end_index]

                assert (len(current) == snippet_length)

                # Append new row to dataframe
                f = h5py.File(event_file_path, "w")
                f.create_dataset('data/block0_values', data=(np.stack((voltage, current), axis=1)))
                f.close()
                os.remove(os.path.join(tmp_path, files[i - 1]))

                # Extend existing csv or create new one
                new_row = {'Medal': medal_id, 'Socket': socket_id, 'Appliance': appliance,
                           'Type': class_name, 'Timestamp': timestamp}
                csv_path = os.path.join(ROOT_DIR, f'data/event_snippets_new/events_new.csv')

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
    labels = pd.read_csv(os.path.join(ROOT_DIR, 'data/events.csv'))

    l = Lock()
    pool = Pool(processes=20, initializer=init, initargs=(l,))

    result = pool.map_async(process_event, labels.iterrows())
    result.get()

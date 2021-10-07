import os
import numpy as np
import h5py
from datetime import timezone
from multiprocessing import Pool
from functools import partial
from datetime import datetime, timedelta


def preprocess_medal(storage_path, path_to_data, date, medal_id):
    """ Preprocesses a day and medal combination and saves the data as numpy array.
        Creates the aggregated Current Over Time.

    Args:
        storage_path (string): Path of the preprocessed files
        path_to_data (string): Path of the dataset
        date (string): String of the date
        medal_id (int): Medal identifier
    """
    medal_path = os.path.join(path_to_data, date, f'medal-{medal_id}')
    file_list = sorted(os.listdir(medal_path))

    for f in file_list[:]:
        if 'summary' in f:
            file_list.remove(f)

    os.makedirs(os.path.join(storage_path, 'tmp'), exist_ok=True)
    pool = Pool(processes=16)
    wrapper = partial(preprocess_file, storage_path, medal_path)
    start_times = pool.map(wrapper, file_list)
    start_times = np.asarray(sorted(start_times))
    pool.close()

    preprocessed_files = sorted(os.listdir(os.path.join(storage_path, 'tmp')))

    # All medal folders should contain 96 files
    assert len(preprocessed_files) == 96

    preprocessed_day = None
    for preprocessed_file in preprocessed_files:
        file_path = os.path.join(storage_path, 'tmp', preprocessed_file)
        file = np.load(file_path)['data']
        preprocessed_day = np.hstack((preprocessed_day, file)) if preprocessed_day is not None else file
        os.remove(file_path)

    storage_path = os.path.join(storage_path, f'{date}_medal-{medal_id}')
    np.savez_compressed(storage_path, data=preprocessed_day, timestamps=start_times)


def preprocess_file(storage_path, path_to_data, file, measurement_frequency=6400, net_frequency=50):
    """

    Args:
        storage_path (string): Path of the preprocessed files
        path_to_data (string): Path of the medal
        file (string): Filename
        net_frequency (int): Frequency of the net 50Hz or 60Hz
        measurement_frequency (int): Frequency of the measurements

    Returns:

    """
    file_path = os.path.join(path_to_data, file)

    data = h5py.File(file_path, 'r')

    start_time = datetime(
        year=int(data.attrs['year']),
        month=int(data.attrs['month']),
        day=int(data.attrs['day']),
        hour=int(data.attrs['hours']),
        minute=int(data.attrs['minutes']),
        second=int(data.attrs['seconds']),
        microsecond=int(data.attrs['microseconds']),
        tzinfo=timezone(timedelta(hours=int(data.attrs['timezone'][1:4]),
                                  minutes=int(data.attrs['timezone'][4:]))))

    cycle_length = int(measurement_frequency / net_frequency)
    cycle_n = int(5760000 / cycle_length)
    np_data = np.empty((6, cycle_n))
    for socket_id in range(1, 7):
        current = data[f'current{socket_id}']
        assert (len(current) == 5760000)
        current -= np.mean(current)
        current = current * data[f'current{socket_id}'].attrs['calibration_factor']
        current = current.reshape(cycle_n, cycle_length)
        current = np.sqrt(np.mean(current ** 2, axis=1))
        np_data[socket_id - 1] = current

    data.close()
    np.savez_compressed(os.path.join(storage_path, 'tmp', file.split('.')[0]), data=np_data)

    return str(start_time)


if __name__ == '__main__':
    path_to_data = "/mnt/nilm/nilm/i13-dataset/BLOND/BLOND-50"
    storage_path = "/mnt/nilm/temp/buchberger/BLOND-Preprocessed"

    start_date = datetime.strptime('2017-01-09', "%Y-%m-%d").date()
    end_date = datetime.strptime('2017-04-30', "%Y-%m-%d").date()
    num_days = (end_date - start_date).days
    dates = [str(start_date + timedelta(days=x)) for x in range(num_days)]
    dates.remove('2017-03-26')
    medal_ids = [7, 10, 11, 12, 13]

    for date in dates:
        for medal_id in medal_ids:
            preprocess_medal(storage_path, path_to_data, date, medal_id)

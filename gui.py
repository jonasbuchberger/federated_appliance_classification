import os
import h5py
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pandas as pd
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pandastable import Table
from multiprocessing import Pool
from functools import partial

PATH_TO_BLOND = "C:/Users/jonas/Documents/MATLAB/BLOND"
PATH_TO_LOG = "C:/Users/jonas/Documents/PyCharm/federated_blond/.old/appliance_log.json"
START_TIME = '2016-10-01T00-00-00'
END_TIME = '2017-05-01T00-00-00'

APPLIANCE_TYPE = {
    'Akura': 'USB Charger',
    'Apple MD836ZM': 'USB Charger',
    'Eurom VS 16': 'Fan',
    'Dell E6540': 'Laptop',
    'Dell Optiplex 7040': 'PC',
    'Dell P2210': 'Monitor',
    'Dell T3600': 'PC',
    'Dell U2711': 'Monitor',
    'Dell U2713Hb': 'Monitor',
    'Dell UP2716D': 'Monitor',
    'Dell XPS13': 'Laptop',
    'Epson EB-65950': 'Projector',
    'FPGA Xilinx ML505': 'Dev Board',
    'Fujitsu-Siemens P17-1': 'Monitor',
    'Heller	ASY 1507': 'Space Heater',
    'HP Laserjet Pro 400': 'Printer',
    'Hama 0091321': 'USB Charger',
    'Kraftmax BC4000 Pro': 'Battery Charger',
    'Lenovo Carbon X1': 'Laptop',
    'Lenovo L540': 'Laptop',
    'Lenovo T420': 'Laptop',
    'Lenovo T450': 'Laptop',
    'Lenovo X230 i5': 'Laptop',
    'Lenovo X230 i7': 'Laptop',
    "MacBook Pro 13'' Mid-2014": 'Laptop',
    "MacBook Pro 15'' Mid-2014": 'Laptop',
    'Philips HF3430': 'Daylight',
    'Projecta DC 485': 'Screen Motor',
    'Prototype': 'Dev Board	',
    'Samsung Travel': 'USB Charger',
    'Schenker W502': 'Laptop',
    'Sony Vaio VGN FW54M': 'Laptop',
    'generic': 'USB Charger',
    'inateck UC2001': 'USB Charger'
}

EVENTS = pd.DataFrame(columns=['Timestamp', 'Medal_Nr', 'Date', 'Socket_Nr', 'Appliance_Name', 'Class_Name'])
root = tk.Tk()

class Graph(tk.Frame):
    def __init__(self, data, medal_id, socket_id, start_time, master=None, *args, **kwargs):
        """ Creates interactive plot for visualization.

        Args:
            data (np.array): Array containing the measurements to plot
            medal_id (int): Medal identifier (1-15)
            socket_id (int): Socket identifier (1-6)
            start_time (datetime): Start time of the day
            master (tk.TK):
            *args: Tkinter arguments
            **kwargs: Tkinter arguments
        """
        super().__init__(master, *args, **kwargs)

        self.medal_id = medal_id
        self.socket_id = socket_id
        self.start_time = start_time
        self.log = get_appliance_start_and_end(medal_id, socket_id)

        self.fig = Figure(figsize=(10, 2))
        ax = self.fig.add_subplot(111)
        ax.grid(True)
        if np.max(data) < 0.5:
            ax.set_ylim([-0.1, 0.5])

        ax.plot(data)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        tk.Label(self, text=f"Graph current{socket_id}").grid(row=0)
        self.canvas.get_tk_widget().grid(row=1, sticky="nesw")
        self.canvas.mpl_connect('button_press_event', self.onclick)
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=2, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def onclick(self, event):
        """ Adds event to dataframe.

        Args:
            event: Event from interactive pyplot
        """
        # If event is from left mouse button
        if event.button == 3 and event.xdata is not None:

            ix, _ = event.xdata, event.ydata

            # Convert click index to timestamp
            ix = ((ix * 128) + 64) / 6400
            timestamp = self.start_time + timedelta(seconds=ix)

            # Get appliance name from log for event timestamp
            appliance = None
            for _, row in self.log.iterrows():
                start = datetime.strptime(row['start'] + '+00:00', "%Y-%m-%dT%H-%M-%S%z")
                end = datetime.strptime(row['end'] + '+00:00', "%Y-%m-%dT%H-%M-%S%z")
                if end > timestamp > start:
                    appliance = row['appliance_name']

            # Try to convert appliance name to class name
            class_name = None
            try:
                class_name = APPLIANCE_TYPE[appliance]
            except KeyError:
                print(appliance)

            row = {'Timestamp': timestamp,
                   'Medal_Nr': self.medal_id,
                   'Date': self.start_time.date(),
                   'Socket_Nr': self.socket_id,
                   'Appliance_Name': appliance,
                   'Class_Name': class_name}

            # Add event to dataframe and update GUI
            global EVENTS
            EVENTS = EVENTS.append(row, ignore_index=True)
            pt.model.df = EVENTS
            pt.redraw()


def get_appliance_start_and_end(medal_id, socket_id):
    """ Returns the start and end time of all appliances of a medal and socket combination

    Args:
        medal_id (int): Medal identifier
        socket_id (int): Index of socket
    Returns:
        df (pd.dataframe): Dataframe containing appliance name, start time, end time
    """

    with open(PATH_TO_LOG) as json_data:
        appliance_log = json.load(json_data)

    medal_log = appliance_log[f'MEDAL-{medal_id}']['entries']

    df = pd.DataFrame(columns=['appliance_name', 'start', 'end'])

    i = 0
    while i < len(medal_log):
        appliance = medal_log[i][f'socket_{socket_id}']['appliance_name']

        start_time = medal_log[i]['timestamp']
        end_time = medal_log[i + 1]['timestamp'] if i < len(medal_log) - 1 else END_TIME

        while i < len(medal_log) - 1 and appliance == medal_log[i + 1][f'socket_{socket_id}']['appliance_name']:
            end_time = medal_log[i + 2]['timestamp'] if i < len(medal_log) - 2 else END_TIME
            i += 1

        start_datetime = datetime.strptime(start_time, "%Y-%m-%dT%H-%M-%S")
        end_datetime = datetime.strptime(end_time, "%Y-%m-%dT%H-%M-%S")

        if start_datetime < end_datetime:
            df = df.append(
                pd.Series([appliance, start_time, end_time],
                          index=df.columns), ignore_index=True)

        i += 1

    return df


def load_data(date, medal_id, socket_id, measurement_frequency=6400, net_frequency=50):
    """

    Args:
        date (string):  Date of the current folder '2016-11-01'
        medal_id (int): Medal identifier (1-15)
        socket_id (int): Socket identifier (1-6)
        net_frequency (int): Frequency of the net 50Hz or 60Hz
        measurement_frequency (int): Frequency of the measurements

    Returns:
        (np.array): Preprocessed current measurements
        (datetime): Start timestamp of day
    """
    cycle_length = int(measurement_frequency / net_frequency)
    medal_path = os.path.join(PATH_TO_BLOND, date, f'medal-{medal_id}')
    files = os.listdir(medal_path)

    start_time = None
    cot_day = None
    for file in sorted(files):
        if 'summary' not in file:
            f = h5py.File(os.path.join(medal_path, file), 'r')

            if start_time is None:
                start_time = datetime(
                    year=int(f.attrs['year']),
                    month=int(f.attrs['month']),
                    day=int(f.attrs['day']),
                    hour=int(f.attrs['hours']),
                    minute=int(f.attrs['minutes']),
                    second=int(f.attrs['seconds']),
                    microsecond=int(f.attrs['microseconds']),
                    tzinfo=timezone(timedelta(hours=int(f.attrs['timezone'][1:4]),
                                              minutes=int(f.attrs['timezone'][4:]))))

            current = process_signal(f, socket_id)
            assert (len(current) == 5760000)

            cycle_n = int(len(current) / cycle_length)
            current = current.reshape(cycle_n, cycle_length)
            current = np.sqrt(np.mean(current ** 2, axis=1))

            cot_day = np.append(cot_day, current) if cot_day is not None else current

    return cot_day, start_time


def process_signal(data, socket_id):
    """ Performs preprocessing on current wave with mean and calibration.

    Args:
        data (h5py.Dataset): Dataset containing the measurements
        socket_id (int): Identifier of the socket to be preprocessed

    Returns:
        (np.array): Array with the preprocessed current
    """
    current = data[f'current{socket_id}']
    current -= np.mean(current)

    current = current * data[f'current{socket_id}'].attrs['calibration_factor']

    return current


def delete_last_event():
    """ Deletes last added event from dataframe
    """
    EVENTS.drop(EVENTS.tail(1).index, inplace=True)
    pt.model.df = EVENTS
    pt.redraw()


def load_day(root, day, medal_id):
    """ Loads next day into GUI
    """
    pool = Pool(processes=6)
    wrapper = partial(load_data, day, medal_id)
    sockets = np.arange(1, 7)
    results = pool.map(wrapper, sockets)

    for i, result in enumerate(results):
        socket_id = i + 1
        current, start_time = result
        Graph(current,
              medal_id=medal_id,
              socket_id=socket_id,
              start_time=start_time,
              master=root,
              width=500).grid(row=(socket_id - 1) // 2, column=(socket_id - 1) % 2)


if __name__ == '__main__':

    root.title('BLOND Annotator: Current Over Time')

    frame = tk.Frame(root)
    frame.grid(row=5, columnspan=2)

    pt = Table(frame, showtoolbar=True, showstatusbar=True, dataframe=EVENTS, width=1400)
    pt.show()

    frame2 = tk.Frame(root)
    frame2.grid(row=4, columnspan=2)
    button1 = tk.Button(frame2, text="Delete Event", command=delete_last_event)
    button1.grid(row=4, column=0)
    e1 = tk.Entry(frame2, bd=5)
    e1.grid(row=4, column=2)
    e1.insert(0, "2016-11-01")
    e2 = tk.Entry(frame2, bd=5)
    e2.grid(row=4, column=3)
    e2.insert(0, "1")
    button2 = tk.Button(frame2, text="Next", command=lambda: load_day(root, e1.get(), int(e2.get())))
    button2.grid(row=4, column=1)

    load_day(root, "2016-11-01", 1)

    root.mainloop()

    # EVENTS.to_csv(str(datetime.now()).replace(':', '_') + '.csv')

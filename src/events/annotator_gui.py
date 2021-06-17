import os
import h5py
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pandastable import Table
from multiprocessing import Pool
from functools import partial
import scipy.signal
import scipy

# Extend this DICT for unknown appliances
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
    'DJI Phantom 3': 'Battery Charger',
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


class SocketGraph(tk.Frame):
    def __init__(self, socket_id, path_to_log, master=None, pandas_table=None,
                 measurement_frequency=6400, net_frequency=50, *args, **kwargs):
        """ Creates empty interactive plot for visualization.

        Args:
            socket_id (int): Socket identifier (1-6)
            path_to_log (string): Path to appliance log
            master (tk.TK): GUI
            pandas_table (pandastable.Table): Dataframe holding events
            net_frequency (int): Frequency of the net 50Hz or 60Hz
            measurement_frequency (int): Frequency of the measurements
            *args: Tkinter arguments
            **kwargs: Tkinter arguments
        """
        super().__init__(master, *args, **kwargs)

        self.medal_id = None
        self.timestamps = None
        self.log = None
        self.socket_id = socket_id
        self.path_to_log = path_to_log
        self.measurement_frequency = measurement_frequency
        self.net_frequency = net_frequency
        self.single_file_length = 5760000
        self.pt = pandas_table

        self.fig = Figure(figsize=(12.8, 3.2))
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        tk.Label(self, text=f"Graph current{socket_id}").grid(row=0)
        self.canvas.get_tk_widget().grid(row=1, sticky="nesw")
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=2, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def draw_graph(self, current, medal_id, timestamps):
        """ Draws new Graph for a socket and updates parameters.

        Args:
            current (np.array): Array containing the measurements to plot
            medal_id (int): Medal identifier (1-15)
            timestamps (list): List of all file start times
        """
        self.medal_id = medal_id
        self.timestamps = timestamps
        self.log = self.get_appliance_start_and_end()
        self.ax.cla()
        self.ax.grid(True)
        self.ax.plot(current, linewidth=0.5)
        if np.max(current) < 0.3:
            self.ax.set_ylim([-0.1, 0.3])
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.draw()

    def onclick(self, event):
        """ Adds event to dataframe and draws red line where clicked.

        Args:
            event: Event from interactive pyplot
        """
        # If event is from left mouse button
        if event.button == 3 and event.xdata is not None:

            cycle_length = int(self.measurement_frequency / self.net_frequency)

            # Get click index of event
            index, _ = event.xdata, event.ydata

            # Find file that includes event start based of the click event
            file_length_processed = self.single_file_length / cycle_length
            file_index = int(index / file_length_processed)
            file_timestamp = self.timestamps[file_index]

            # Calculate the offset of the event in the correct file in seconds
            file_offset = (index % file_length_processed) * cycle_length / self.measurement_frequency

            # Event timestamp is start time of file + offset
            timestamp = file_timestamp + timedelta(seconds=file_offset)

            # Get appliance name from log for event timestamp
            appliance = None
            for _, row in self.log.iterrows():
                start = datetime.strptime(str(row['start']) + '+0000', "%Y-%m-%dT%H-%M-%S%z")
                end = datetime.strptime(str(row['end']) + '+0000', "%Y-%m-%dT%H-%M-%S%z")
                if end > timestamp > start:
                    appliance = row['appliance_name']

            # Try to convert appliance name to class name
            # Class name can be added manually in GUI
            class_name = None
            try:
                class_name = APPLIANCE_TYPE[appliance]
            except KeyError:
                print(appliance)

            row = {'Timestamp': timestamp,
                   'Medal_Nr': self.medal_id,
                   'Date': timestamp.date(),
                   'Socket_Nr': self.socket_id,
                   'Appliance_Name': appliance,
                   'Class_Name': class_name}

            # Add event to dataframe and update GUI
            global EVENTS
            EVENTS = EVENTS.append(row, ignore_index=True)
            self.pt.model.df = EVENTS
            self.pt.redraw()

            self.ax.axvline(x=index, color='r')
            self.fig.canvas.draw()

    def get_appliance_start_and_end(self):
        """ Returns the start and end time of all appliances of a medal and socket combination.

        Returns:
            df (pd.dataframe): Dataframe containing appliance name, start time, end time
        """

        end_time = '2017-05-01T00-00-00'

        with open(self.path_to_log) as json_data:
            appliance_log = json.load(json_data)

        medal_log = appliance_log[f'MEDAL-{self.medal_id}']['entries']

        df = pd.DataFrame(columns=['appliance_name', 'start', 'end'])

        i = 0
        while i < len(medal_log):
            appliance = medal_log[i][f'socket_{self.socket_id}']['appliance_name']

            start_time = medal_log[i]['timestamp']
            end_time = medal_log[i + 1]['timestamp'] if i < len(medal_log) - 1 else end_time

            while i < len(medal_log) - 1 and appliance == medal_log[i + 1][f'socket_{self.socket_id}'][
                'appliance_name']:
                end_time = medal_log[i + 2]['timestamp'] if i < len(medal_log) - 2 else end_time
                i += 1

            start_datetime = datetime.strptime(start_time, "%Y-%m-%dT%H-%M-%S")
            end_datetime = datetime.strptime(end_time, "%Y-%m-%dT%H-%M-%S")

            if start_datetime < end_datetime:
                df = df.append(
                    pd.Series([appliance, start_time, end_time],
                              index=df.columns), ignore_index=True)

            i += 1

        return df


class Annotator:
    def __init__(self, path_to_data, path_to_log, path_to_preprocessed='', measurement_frequency=6400,
                 net_frequency=50):
        """ Creates an Annotator GUI for the BLOND dataset.
            The plots show the aggregated Current Over Time of appliances.
            Events are created with right mouse click on the plots.
            For each plot you can zoom and drag the graph individually to find events.
            Recommended to zone as close as possible to increase accuracy of events (x-axis shows 1 digits ints).

            Preprocess:
                There is the ability to preprocess day medal combinations to reduce loading time.
                Specify days and medals in preprocess.py and add the correct path to the initialization.

            GUI:
                Delete Last Event: Deletes last event in the event dataframe
                Load: Load the date medal combination specified in the two text fields to the right
                Load CSV: Load an already created events.csv
                Save CSV: Save current events as .csv file

        Args:
            path_to_data (string): Path to BLOND-50 dataset
            path_to_log (string): Path to appliance log
            path_to_preprocessed (string): Path to preprocessed data for faster loading
            measurement_frequency (int):
            net_frequency (int):
        """
        self.path_to_data = path_to_data
        self.path_to_log = path_to_log
        self.path_to_preprocessed = path_to_preprocessed
        self.measurement_frequency = measurement_frequency
        self.net_frequency = net_frequency

        self.cache = {}
        self.graphs = []
        self.root = tk.Tk()
        self.root.title('BLOND Annotator: Current Over Time')

        self.frame = tk.Frame(self.root)
        self.frame.pack(fill='both', expand=True)
        self.frame.grid(row=5, columnspan=2)
        self.pt = Table(self.frame, width=1500, height=150)
        self.pt.show()

        frame2 = tk.Frame(self.root)
        frame2.grid(row=4, columnspan=2)
        button1 = tk.Button(frame2, text="Delete Last Event", command=self.delete_last_event)
        button1.grid(row=4, column=0)
        e1 = tk.Entry(frame2, bd=5)
        e1.grid(row=4, column=3)
        e1.insert(0, "2016-10-01")
        e2 = tk.Entry(frame2, bd=5)
        e2.grid(row=4, column=4)
        e2.insert(0, "1")
        button2 = tk.Button(frame2, text="Load", command=lambda: self.load_day(e1.get(), int(e2.get())))
        button2.grid(row=4, column=1)
        button3 = tk.Button(frame2, text='Save CSV', command=lambda: self.save_events_to_csv())
        button3.grid(row=4, column=5)
        button4 = tk.Button(frame2, text='Load CSV', command=lambda: self.load_events_csv())
        button4.grid(row=4, column=6)

        # Setup empty graph plots
        for i in range(0, 6):
            self.graphs.append(SocketGraph(socket_id=i + 1,
                                           path_to_log=self.path_to_log,
                                           master=self.root,
                                           pandas_table=self.pt))
            self.graphs[i].grid(row=i // 2, column=i % 2)

        self.root.mainloop()

    def load_day(self, day, medal_id):
        """ Loads next day into GUI.

        Args:
            day (string): Day to load
            medal_id (int): Medal identifier
        """
        sockets = np.arange(1, 7)
        preprocessed_path = os.path.join(self.path_to_preprocessed, f'{day}_medal-{medal_id}.npz')

        # Check if day is available preprocessed
        if os.path.isfile(preprocessed_path):
            processed_data = np.load(preprocessed_path)['data']
            timestamps = np.load(preprocessed_path)['timestamps']

            # Convert string timestamps to datetimes
            timestamps = [datetime.fromisoformat(date) for date in timestamps]
            results = []
            for i in range(0, 6):
                results.append((processed_data[i], timestamps))

        # Else process day, takes longer to load
        else:
            pool = Pool(processes=6)
            wrapper = partial(load_data, self.path_to_data, day, medal_id)
            results = pool.map(wrapper, sockets)
            pool.close()

        # Update graphs
        for i, result in enumerate(results):
            current, timestamps = result
            self.graphs[i].draw_graph(current, medal_id, timestamps)

    def save_events_to_csv(self):
        save_path = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        EVENTS.to_csv(save_path, mode='w', encoding='utf-8', line_terminator='\n')

    def load_events_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        global EVENTS
        EVENTS = pd.read_csv(file_path, index_col=0)
        self.pt.model.df = EVENTS
        self.pt.redraw()

    def delete_last_event(self):
        """ Deletes last added event from dataframe
        """
        EVENTS.drop(EVENTS.tail(1).index, inplace=True)
        self.pt.model.df = EVENTS
        self.pt.redraw()


def load_data(path_to_data, date, medal_id, socket_id, measurement_frequency=6400, net_frequency=50):
    """ Loads and preprocesses a day and medal.
        Uses one thread for each socket to decrease loading times.

    Args:
        path_to_data (string): Path to the dataset
        date (string):  Date of the current folder '2016-11-01'
        medal_id (int): Medal identifier (1-15)
        socket_id (int): Socket identifier (1-6)
        net_frequency (int): Frequency of the net 50Hz or 60Hz
        measurement_frequency (int): Frequency of the measurements

    Returns:
        (np.array): Preprocessed current measurements
        (list): Start timestamp of each file
    """
    cycle_length = int(measurement_frequency / net_frequency)
    medal_path = os.path.join(path_to_data, date, f'medal-{medal_id}')
    files = os.listdir(medal_path)

    timestamps = []
    cot_day = None
    for file in sorted(files):
        if 'summary' not in file:
            f = h5py.File(os.path.join(medal_path, file), 'r')

            time = datetime(
                year=int(f.attrs['year']),
                month=int(f.attrs['month']),
                day=int(f.attrs['day']),
                hour=int(f.attrs['hours']),
                minute=int(f.attrs['minutes']),
                second=int(f.attrs['seconds']),
                microsecond=int(f.attrs['microseconds']),
                tzinfo=timezone(timedelta(hours=int(f.attrs['timezone'][1:4]),
                                          minutes=int(f.attrs['timezone'][4:]))))
            timestamps.append(time)

            current = process_signal(f, socket_id)

            # Assert no missing values in file
            assert len(current) == 5760000

            cycle_n = int(len(current) / cycle_length)
            current = current.reshape(cycle_n, cycle_length)
            current = np.sqrt(np.mean(current ** 2, axis=1))

            cot_day = np.append(cot_day, current) if cot_day is not None else current

    return cot_day, timestamps


def process_signal(data, socket_id):
    """ Performs preprocessing on current wave with mean and calibration.
        For loading performance the median filter and offset correction is not applied.
        For finding events this should not be necessary.

    Args:
        data (h5py.File): Dataset containing the measurements
        socket_id (int): Identifier of the socket to be preprocessed

    Returns:
        (np.array): Array with the preprocessed current
    """
    # _, offset_current = calibrate_offset(data, data.attrs['frequency'])
    current = data[f'current{socket_id}']
    # current -= offset_current
    current -= np.mean(current)
    # current = scipy.signal.medfilt(current, 15)

    current = current * data[f'current{socket_id}'].attrs['calibration_factor']

    return current


def calibrate_offset(data, average_frequency):
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


if __name__ == '__main__':
    # path_to_data = "C:/Users/jonas/Documents/MATLAB/BLOND"
    # path_to_log = "C:/Users/jonas/Documents/PyCharm/federated_blond/.old/appliance_log.json"
    # path_to_preprocessed = "C:/Users/jonas/Documents/MATLAB/BLOND-Preprocessed"

    path_to_data = "/mnt/nilm/nilm/i13-dataset/BLOND/BLOND-50"
    path_to_log = "/mnt/nilm/nilm/i13-dataset/BLOND/appliance_log.json"
    path_to_preprocessed = "/mnt/nilm/temp/buchberger/BLOND-Preprocessed"

    annotator = Annotator(path_to_data=path_to_data,
                          path_to_log=path_to_log,
                          path_to_preprocessed=path_to_preprocessed)

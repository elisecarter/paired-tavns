from asyncio import threads
from psychopy import core, sound, prefs
import nidaqmx
from nidaqmx import stream_readers, constants
from nidaqmx.constants import AcquisitionType, FrequencyUnits, LineGrouping
from pupil_labs.realtime_api.simple import Device
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import numpy as np
import pandas as pd
import threading
import os
import time
import json


class DataCollector:
    """
    data storage and thread-related variables
    """
    def __init__(self):
        self.event_data = []
        self.pupil_data = []
        self.bitalino_data = []
        self.bitalino_channels = {}
        self.running = True

        # self.ni_daq_lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.pupil_lock = threading.Lock()
        self.bitalino_lock = threading.Lock()


def initialize_stimulation_task(device, stim_freq, onTime):
    counter_channel = f'{device}/ctr0'
    pulse_width = 10e-6  # Pulse width in seconds (10 microseconds)
    pulse_no = int(stim_freq * onTime)  # Number of pulses in the train for onTime duration
    duty_cycle = pulse_width * stim_freq  # Duty cycle to achieve 10 microseconds high
    stim_task = nidaqmx.Task()
    stim_task.co_channels.add_co_pulse_chan_freq(
        counter_channel,
        units=FrequencyUnits.HZ,
        freq=stim_freq,
        duty_cycle=duty_cycle
    )
    stim_task.timing.cfg_implicit_timing(
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=pulse_no
    )

    return stim_task

def trigger_codes_to_bool_list(trigger_codes):
    """
    Converts each trigger code to a boolean list.
    """
    for key, value in trigger_codes.items():
        trigger_codes[key] = [(value >> bit) & 1 == 1 for bit in range(3)]  # Assuming 3 bits for the trigger codes
    
    return trigger_codes

def initialize_trigger_code_task(device, trigger_codes):
    """ Initialize the digital output task for trigger codes """
    # Convert trigger codes to boolean lists
    trigger_codes_bool = trigger_codes_to_bool_list(trigger_codes)

    # Create a digital output task
    code_output_task = nidaqmx.Task()
    code_output_task.do_channels.add_do_chan(
        f"{device}/port1/line1:3", 
        line_grouping=LineGrouping.CHAN_PER_LINE)
    
    return code_output_task, trigger_codes_bool

def write_trigger_code(task, bool_list, duration, num_bits=3):
    """
    Write a trigger code to the digital output task (NON-BLOCKING)
    """
    def code_thread():
        task.write(bool_list, auto_start=True)
        time.sleep(duration)
        task.write([False] * num_bits, auto_start=True)
    
    threading.Thread(target=code_thread).start()

def initialize_rec_task(device, rec_f, num_channels, frames_per_buffer, ni_daq_outlet):
    ai_task = nidaqmx.Task()
    for i in range(num_channels):
        ai_task.ai_channels.add_ai_voltage_chan(
            f"{device}/ai{i}",
            min_val=-10.0,
            max_val=10.0
        )

    ai_task.timing.cfg_samp_clk_timing(
        rate=rec_f,
        sample_mode=AcquisitionType.CONTINUOUS
    )
    #ai_task.input_buf_size = frames_per_buffer * num_channels
    refresh_rate_hz = rec_f / frames_per_buffer
    samples_per_frame = int(rec_f // refresh_rate_hz)

    # Create a multi-channel reader and buffer
    reader = stream_readers.AnalogMultiChannelReader(ai_task.in_stream)
    read_buffer = np.zeros((num_channels, samples_per_frame), dtype=np.float64)

    def reading_task_callback(task_idx, event_type, num_samples, callback_data=None):
        # Read samples into buffer
        reader.read_many_sample(
            read_buffer,
            num_samples,
            timeout=constants.WAIT_INFINITELY
        )
        # Push samples to LSL
        data_to_push = read_buffer.T.tolist()
        ni_daq_outlet.push_chunk(data_to_push)
        return 0

    # Register the callback to run every n samples
    ai_task.register_every_n_samples_acquired_into_buffer_event(
        samples_per_frame, 
        reading_task_callback
    )

    ai_task.start()
    return ai_task

def setup_lsl_outlets():
    """ Set up LSL streams for event timestamps """
    # event markers outlet
    event_info = StreamInfo(
        name='Event_Stream',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string', 
        source_id='event12345'
    )
    event_outlet = StreamOutlet(event_info)

    return event_outlet

def _to_host_time(timestamps, inlet):
    """Convert device timestamps to host time using a single contemporaneous time_correction.
    Returns (ts_host, corr_used)."""
    if hasattr(inlet, "estimate_time_offset"):
        corr = inlet.estimate_time_offset().time_offset_ms.mean / 1000.0
    elif hasattr(inlet, "time_correction"):
        corr = inlet.time_correction()
    else:
        corr = 0.0
    return [t + corr for t in timestamps], corr

def receive_pupil_lsl(data_collector):
    # Resolve the Pupil Labs stream 
    print("Looking for Pupil Labs data stream...")
    pupil_streams = resolve_byprop('name', 'Neon Companion_Neon Gaze',timeout=5)
    if not pupil_streams:
        print("Pupil Labs stream not found.")
        return
    pupil_inlet = StreamInlet(pupil_streams[0])
    pupil_inlet.open_stream(timeout=1.0)
    print("Pupil Labs stream found.")
    while data_collector.running:
        samples, timestamps = pupil_inlet.pull_chunk(timeout=1.0)
        if samples and timestamps:
            ts_host, corr = _to_host_time(timestamps, pupil_inlet)
            with data_collector.pupil_lock:
                for s, t in zip(samples, ts_host):
                    data_collector.pupil_data.append({
                        'Timestamp': t, 
                        'Pupil_Diameter': s[2], 
                        'TimeCorrection': corr})

def receive_bitalino_lsl(data_collector):
    # Resolve the Bitalino stream
    mac_add = "20:19:07:00:80:63"
    print("Looking for Bitalino data stream...")
    stream = resolve_byprop('type', mac_add, timeout=5.0)
    if not stream:
        print("Bitalino stream not found.")
        return
    inlet = StreamInlet(stream[0])
    print("Bitalino stream found.")
    # Access the channel number and data type
    stream_info = inlet.info()
    channel_number = stream_info.channel_count()
    # Store sensor channel info & units in the dictionary
    channels = stream_info.desc().child("channels").child("channel")
    # Loop through all available channels
    stream_channels = dict()
    for i in range(channel_number):
        # Get the channel number (e.g. 1)
        channel = i + 1
        # Get the channel type (e.g. ECG)
        sensor = channels.child_value("label")
        # Get the channel unit (e.g. mV)
        unit = channels.child_value("unit")
        # Store the information in the stream_channels dictionary
        stream_channels.update({channel: [sensor, unit]})
        channels = channels.next_sibling()
    data_collector.bitalino_channels = stream_channels
    
    inlet.open_stream(timeout=1.0)
    while data_collector.running:
        samples, timestamps = inlet.pull_chunk(timeout=1.0)
        if samples and timestamps:
            ts_host, corr = _to_host_time(timestamps, inlet)
            with data_collector.bitalino_lock:
                for s, t in zip(samples, ts_host):
                    data_collector.bitalino_data.append({
                        'Timestamp': t,
                        'Sample': s,
                        'TimeCorrection': corr
                    })

def receive_events_lsl(data_collector):
    print("Looking for events stream...")
    stroop_streams = resolve_byprop('name', 'Event_Stream')
    if not stroop_streams:
        print("Events stream not found.")
        return
    event_inlet = StreamInlet(stroop_streams[0])
    event_inlet.open_stream(timeout=1.0)
    print("Events stream found.")
    while data_collector.running:
        sample, timestamp = event_inlet.pull_sample(timeout=1.0)
        if sample and timestamp:
            ts_host, corr = _to_host_time([timestamp], event_inlet)
            with data_collector.event_lock:
                data_collector.event_data.append({
                    'Timestamp': ts_host[0],
                    'Event': sample[0],
                    'TimeCorrection': corr
                })

# ---- DIRECT MODE: Pupil via Pupil Labs Realtime API ----
def receive_pupil_direct(neon, data_collector, host="127.0.0.1", port=8080):
    """
    Subscribe to Pupil Labs realtime API and append diameter samples to data_collector.pupil_data
    Schema matches LSL path: {'Timestamp', 'Pupil_Diameter', 'LSL_TimeCorrection'}
    """ 
    # Example skeleton; fill with your exact API calls (device.gaze() or device.pupil())
    from pupil_labs.realtime_api.simple import Device
    try:
        while data_collector.running:
            data = neon.receive_gaze_datum(timeout_seconds=1.0)
            diam_left = getattr(data, "pupil_diameter_left", None)  # adjust to your API
            diam_right = getattr(data, "pupil_diameter_right", None)  # adjust to your API
            timestamps = getattr(data, "timestamp_unix_seconds", None)
            if diam_left is None or diam_right is None:
                    continue
            ts_host, corr = _to_host_time([timestamps], neon)
            with data_collector.pupil_lock:
                data_collector.pupil_data.append({
                    'Timestamp': ts_host[0],
                    'Sample': tuple([diam_left, diam_right]),
                    'TimeCorrection': corr
                })
    except Exception as e:
        print(f"Pupil direct acquisition error: {e}")


# ---- DIRECT MODE: BITalino via Bluetooth (python-bitalino) ----
try:
    from bitalino import BITalino
except Exception:
    BITalino = None

def receive_bitalino_direct(data_collector, mac, srate=1000, channels=[0,1,2,3,4,5], chunk=15):
    if BITalino is None:
        print("bitalino package not installed.")
        return
    dev = None
    try:
        dev = BITalino(mac)
        dev.start(srate, channels)
        # mirror LSL channel file structure: {1: ["ECG","mV"], ...}
        data_collector.bitalino_channels = {i+1: [f"A{i+1}", "au"] for i in range(len(channels))}
        data_collector.bitalino_channels.update({0: ["nSeq",""]})
        # first time we see data, build channel metadata (like your LSL path does)
        while data_collector.running:
            arr = dev.read(chunk)  # rows: [seq, A1..An, D1..D4]
            print(arr)
            t_recv = time.monotonic()
            if arr is None or len(arr) == 0:
                continue
            with data_collector.bitalino_lock:
                for row in arr:
                    seq = int(row[0])
                    analog = row[5:5+len(channels)].tolist()
                    data_collector.bitalino_data.append({
                        "Timestamp": t_recv,      # host monotonic
                        "Sample": analog,         # same field as LSL path
                        "TimeCorrection": 0.0,
                        "nSeq": seq
                    })
    except Exception as e:
        print(f"BITalino direct acquisition error: {e}")
    finally:
        try:
            if dev is not None:
                dev.stop(); dev.close()
        except Exception:
            pass

def start_direct_collection_threads(self):
    threads = []
    if self.record_pupil:
        threads.append(threading.Thread(target=receive_pupil_direct, args=(self.neon, self.data_collector, self.neon_ip, 8080)))
    if self.record_bitalino:
        mac = self.config.get("bitalino_mac")
        srate = self.config.get("bitalino_srate", 1000)
        chs = [0,1,2,3,4,5]
        threads.append(threading.Thread(target=receive_bitalino_direct, args=(self.data_collector, mac, srate, chs)))
    for t in threads: t.start()
    return tuple(threads)
                
def start_lsl_collection_threads(self):
    threads = []
    threads.append(threading.Thread(target=receive_events_lsl, args=(self.data_collector,)))
    if self.record_pupil:
        threads.append(threading.Thread(target=receive_pupil_lsl, args=(self.data_collector,)))
    if self.record_bitalino:
        threads.append(threading.Thread(target=receive_bitalino_lsl, args=(self.data_collector,)))

    for t in threads: t.start()
    return tuple(threads)

def stop_data_collection_threads(threads, data_collector):
    data_collector.running = False
    for thread in threads:
        if thread.is_alive():
            thread.join()

def save_data(ID, datetime, data_collector,config):
    # Create data directories & paths
    date_str = datetime.strftime("%Y%m%d")
    datetime_str = datetime.strftime("%Y%m%d-%H%M")
    data_dir = os.path.join("Data", ID, date_str, datetime_str)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Save config file to json
    config_path = os.path.join(data_dir, f"{datetime_str}_config.json")
    with open(config_path, "w") as outfile:
        config["datetime"] = datetime_str
        json.dump(config, outfile, indent=4)

    # Convert lists to DataFrames
    # ni_daq_df = pd.DataFrame(data_collector.ni_daq_data)
    event_df = pd.DataFrame(data_collector.event_data)
    pupil_df = pd.DataFrame(data_collector.pupil_data)
    bitalino_df = pd.DataFrame(data_collector.bitalino_data)
    channels_info = data_collector.bitalino_channels

    # Set 'Timestamp' as the index if available and save to csv
    # if not ni_daq_df.empty:
    #     ni_daq_df.set_index('Timestamp', inplace=True)
    #     ni_daq_df.to_csv(os.path.join(data_dir, f"{ID}_daq.csv"))
    if not event_df.empty:
        event_df.set_index('Timestamp', inplace=True)
        event_df.to_csv(os.path.join(data_dir, f"{datetime_str}_events.csv"))
    if not pupil_df.empty:
        pupil_df.set_index('Timestamp', inplace=True)
        pupil_df.to_csv(os.path.join(data_dir, f"{datetime_str}_pupil.csv"))
    if not bitalino_df.empty:
        bitalino_df.set_index('Timestamp', inplace=True)
        bitalino_df.to_csv(os.path.join(data_dir, f"{datetime_str}_bitalino.csv"))
        # Save config file to json
        path = os.path.join(data_dir, f"{datetime_str}_channels.json")
        with open(path, "w") as outfile:
            json.dump(channels_info, outfile, indent=4)
    
    print(f"Data saved in {data_dir}")

    
class Experiment:
    """    
    Experiment class to handle the setup and execution of data acquisitiuon.
    """
    def __init__(self, config):
        # Configuration parameters (required and optional)
        self.ID = config["ID"]
        self.datetime = config["datetime"]
        self.condition = config["condition"]
        self.trigger_stim = config["trigger_stim"]
        self.send_trigger_codes = config["send_trigger_codes"]
        self.trigger_codes = config["trigger_codes"]
        self.record_pupil = config["record_pupil"]
        self.record_bitalino = config["record_bitalino"]
        self.acquisition_mode = config["acquisition_mode"]
        self.stim_freq = config["stim_freq"]
        self.stim_dur = config["stim_dur"]
        self.neon_ip = config["neon_ip"]
        self.bitalino_mac = config["bitalino_mac"]
        self.bitalino_srate = config["bitalino_srate"]
        self.bitalino_ch = config["bitalino_channels"]
        self.data_collector = DataCollector()
        self.config = config
        self.correct_sound = sound.Sound(os.path.join(os.path.dirname(os.path.realpath(__file__)), "correctresponse.wav"))
        self.incorrect_sound = sound.Sound(os.path.join(os.path.dirname(os.path.realpath(__file__)), "wrongresponse.wav"))
        self.trigger_code_task = None

    def setup_data_streams(self):
        # Pupil: start Neon recording to LSL
        if self.record_pupil:
            print("Connecting to Pupil Labs device...")
            self.neon = Device(address=self.neon_ip, port=8080)
            self.recording_id = self.neon.recording_start()
            print(f"Pupil labs recording {self.recording_id} initiated.")
        else:
            self.neon = None
            print("Pupil Labs recording disabled.")

        if self.acquisition_mode == "LSL":
            # Events come from LSL → we publish Event_Stream.
            self.event_outlet = setup_lsl_outlets()
            print("Streaming task events to LSL.")
            # Start LSL collection threads (pupil/events/bitalino)
            self.threads = start_lsl_collection_threads(self)
            print("LSL data collection threads started.")

        else:  # acquisition_mode == "direct"
            self.event_outlet = None
            print("Direct mode: LSL disabled. Events and data will use host clock.")
            self.threads = start_direct_collection_threads(self)
            print("Direct data collection threads started.")

            
    def setup_stimulation_trigger(self):
        if self.trigger_stim:
            # Initialize stimulation task
            self.stim_task= initialize_stimulation_task('Dev1', self.stim_freq, self.stim_dur)
            print("Stimulation task initialized.")
        else:
            self.stim_task = None
            print("Stimulation task disabled.")

    def log_event(self, label: str):
        t_host = time.monotonic()  # seconds, monotonic system clock
        if self.acquisition_mode == "lsl" and self.event_outlet is not None:
            self.event_outlet.push_sample([label])  # primary clock is LSL
        else:  # Always keep a local copy with host time for redundancy/QA
            with self.data_collector.event_lock:
                self.data_collector.event_data.append({
                    "Timestamp": t_host,
                    "Event": label,
                    "TimeCorrection": 0.0
            })
        
    def send_trigger(self, trigger_label, duration=0.1):
        if self.trigger_code_task is not None:
            # Get the trigger code for the specified label  
            bool = self.trigger_codes[trigger_label]
            write_trigger_code(self.trigger_code_task, bool, duration=duration)
            print(f"Trigger code '{trigger_label}' sent.")
        else:
            print("Trigger code task not initialized.")

    def stop_threads(self):
        if self.threads is not None:
            stop_data_collection_threads(self.threads, self.data_collector)
            print("Data collection threads closed")

    def save_data(self):
        print('Saving data...')
        save_data(self.ID, self.datetime, self.data_collector, self.config)

    def cleanup(self):
        if self.stim_task is not None:
            self.stim_task.close()
            print('Stim task successfully closed')
        if self.trigger_code_task is not None:
            self.trigger_code_task.close()
            print('Trigger code task successfully closed')
        if self.neon is not None:
            self.neon.recording_stop_and_save()
            time.sleep(1)  # Give some time for the recording to save
            self.neon.close()
            print('Neon recording successfully closed')
    
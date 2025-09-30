import os
import pandas as pd
import json
import numpy as np
import scipy.signal as signal
import argparse
import sys
import traceback
from based_noise_blinks_detection import detect_blinks
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pointprocess_py import compute_full_regression

def preprocess_pupil_data(pupil_df):
    """
    Smooth pupil diameter and interpolate missing values due to blinks.

    Returns:
        dict: {
            "signal_type": "PUPILDIAM",
            "unit": "mm",
            "Fs": float,      # Sampling frequency (Hz) estimated from average timestamp difference
            "Fs_SD": float,   # Standard deviation of timestamp differences (Hz)
            "timestamps": list,
            "data": list
        }
    """
    if pupil_df is None or pupil_df.empty:
        return None
    
    # sort timestamps
    pupil_df = pupil_df.drop_duplicates(subset='Timestamp', keep='first') 
    pupil_df = pupil_df.sort_values(by='Timestamp').reset_index(drop=True)
    
    dt = np.diff(pupil_df['Timestamp'])
    avg_dt = round(np.mean(dt), 3)  # average time difference between timestamps
    # plot raw pupil data
    # plt.figure(figsize=(12, 6))
    # plt.plot(pupil_df['Timestamp'], pupil_df['Pupil_Diameter'], label='raw pupil')
    
    # Detect blink artifacts
    blinks = detect_blinks(pupil_df['Pupil_Diameter'], sampling_freq=1/avg_dt)
    for onset, offset in zip(blinks['blink_onset'], blinks['blink_offset']):
        pupil_df.loc[int(onset):int(offset), 'Pupil_Diameter'] = np.nan
    pupil_df['interpolated'] = pupil_df['Pupil_Diameter'].isnull()
    
    # Interpolate through blinks
    x = pupil_df['Timestamp'].values
    y = pupil_df['Pupil_Diameter'].values
    nans = np.isnan(y)
    pupil_df['Pupil_Diameter'] = np.interp(x, x[~nans], y[~nans]) #seems more natural than PchipInterpolator
    
    # resample to uniform 200 Hz
    t_start = x[0]
    t_end = x[-1]
    dt = 1/200  # target sampling interval for 200 Hz
    uniform_t = np.arange(t_start, t_end, dt).round(3)
    interp_y = np.interp(uniform_t, x, pupil_df['Pupil_Diameter'].values)
    pupil_df = pd.DataFrame({
        'Timestamp': uniform_t,
        'Pupil_Diameter': interp_y
    })
    
    # plot interpolated pupil data
    # plt.plot(pupil_df['Timestamp'], pupil_df['Pupil_Diameter'], label='interp pupil')

    # filter pupil diameter using a low-pass filter
    filt = signal.butter(4, 6, fs=1/dt, btype='low', output='sos')
    pupil_df['Pupil_Diameter'] = signal.sosfiltfilt(filt, pupil_df['Pupil_Diameter'])
    
    
    
    # plot filtered pupil data
    # plt.plot(pupil_df['Timestamp'], pupil_df['Pupil_Diameter'], label='filtered')
    # plt.show()

    if not pupil_df.empty:
        preprocessed_data = {
            "signal_type": "PUPILDIAM",
            "unit": "mm",
            "Fs": 1/dt,
            "timestamps": pupil_df['Timestamp'].tolist(),
            "data": pupil_df['Pupil_Diameter'].tolist()
        }
         
    return preprocessed_data

def preprocess_daq_data(daq_df, channel_info):
    """Parse channel names from channel file and filter data based on signal type."""
    if daq_df is None or daq_df.empty:
        return None

    dt = 0.001

    # Create mapping from column number to signal type (first entry of the list)
    channel_types = {int(ch): info[0] for ch, info in channel_info.items()}
    # iterate through columns in daq_df['Sample'] and process based on channel type
    preprocessed_data = []
    for col, signal_type in channel_types.items():

        if 'nSeq' in signal_type:
            unit = ''
                # pull column data from daq_df['Sample']
            dat = daq_df['Sample'].apply(lambda x: x.strip("[]").split(',')[col-1] if isinstance(x, str) else None)
            dat = pd.to_numeric(dat)
            nSeq = dat.values
            # Vectorized timestamp calculation
            diff = np.diff(nSeq, prepend=nSeq[0])
            diff[nSeq == 0] += 16
            t = np.round(daq_df['Timestamp'][0] + np.cumsum(diff * dt), 3)
            daq_df['Timestamp'] = t
            dat = nSeq
            
    # Then process all other signal types
    for col, signal_type in channel_types.items():
        if 'nSeq' in signal_type:
            continue  # already processed above
        dat = daq_df['Sample'].apply(lambda x: x.strip("[]").split(',')[col-1] if isinstance(x, str) else None)
        dat = pd.to_numeric(dat)
        t = daq_df['Timestamp']
    
        if 'ECG' in signal_type:
            signal_type = 'IBI'
            unit = 's'
            
            fs = 1/dt
            window_size = 0.03  # seconds
            ecg = pd.Series(dat, dtype=float)
            
            # bandpass filter ECG
            filt = signal.butter(4, [0.5, 30], fs=1/dt, btype='band', output='sos')
            vFilt = signal.sosfiltfilt(filt, ecg)
            
            # take derivative and square
            dV = np.gradient(vFilt, dt)  # compute derivative using numpy gradient
            dV2 = dV ** 2  # square the derivative

            # moving-window integration
            N = int(window_size / dt)
            kernel = np.ones(N) / N
            vInt = np.convolve(dV2, kernel, mode='same')
            
            # find peaks in integrated signal
            from scipy.stats import iqr
            mpd = int(0.2 * fs)  # minimum peak distance in samples
            mph = np.mean(vInt) + 1 * np.std(vInt)  # minimum peak height

            peaks, _ = signal.find_peaks(vInt, distance=mpd, height=mph)
                    
            corrected_peaks = []
            window_dur = 0.01 # seconds
            half_window = int(window_dur / dt)  # number of samples in the window
            for peak in peaks:
                if peak < half_window or peak > len(ecg) - half_window - 1:
                    continue
                window = ecg[peak - half_window: peak + half_window + 1].values
                corrected_peak = peak - half_window + np.argmax(np.array(window))
                corrected_peaks.append(corrected_peak)
            corrected_peaks = pd.Series(corrected_peaks).astype(int)

            # plt.figure(figsize=(12, 6))
            # plt.plot(t, ecg, label=f'raw ecg')
            # plt.scatter(beatTimes, ecg[nn_peaks], color='r', marker='x', label='peaks')
            # plt.legend()
            # plt.show()

            # launch editor
            block_path = daq_df['block_path'][0]
            peaks_path = os.path.join(block_path, "corrected_peaks.npy")
            manual_correction = True
            if manual_correction:
                corrected_peaks = launch_peak_editor(t.values, ecg.values, corrected_peaks, block_path)
            elif os.path.exists(peaks_path):
                corrected_peaks = np.load(peaks_path).tolist()
            nn_peaks = corrected_peaks
            beatTimes = t.iloc[nn_peaks].values
            ibi = np.diff(beatTimes)  # inter-beat intervals in milliseconds
            dat = ibi
            t = beatTimes[1:]
            
            
            # Compute the instantaneous HRV series with the right-edge window.
            res = compute_full_regression(
                events=beatTimes,
                window_length=30,
                delta=0.5,
                ar_order=2,
                alpha=0.05,
                max_iter=500,
            )

            d = res.to_dict()
            sd_rr = d["sd_RR"]
            
            
            # roll = pd.Series(dat).rolling(int(30/np.median(dat)), ).std()  # crude approx
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            # # Top axis: IBI and estimated mean RR
            # ax1.plot(t, dat, "r.", label='IBI')
            # ax1.plot(d["Time"], d["Mu"], label="Estimated μ (mean RR)")
            # ax1.set_ylabel("IBI / μ (s)")
            # ax1.legend(loc="upper right")
            # # Bottom axis: estimated and empirical SD
            # ax2.plot(d["Time"], sd_rr, label="Estimated σ_RR (std RR)")
            # ax2.plot(t[:len(roll)], roll, alpha=0.5, label="Empirical SD")
            # ax2.set_ylabel("SD_RR (s)")
            # ax2.set_xlabel("Time (s)")
            # ax2.legend(loc="upper right")
            # plt.tight_layout()
            # plt.show()
            
                        
            preprocessed_data.append({
            "signal_type": "SD_RR",
            "unit": "ms",
            "Timestamps": d["Time"].tolist(),
            "data": sd_rr.tolist() if hasattr(sd_rr, "tolist") else list(sd_rr)
            })
            # pass

        elif 'EDA' in signal_type:
            signal_type = 'SCR'
            unit = 'μS'
            eda = dat
            eda = eda.rolling(window=int(0.01/dt), center=True).median()
            eda = eda.bfill().ffill()  # ensure same length as original eda
            # plt.figure(figsize=(12, 6))
            # plt.plot(t, eda-20, label=f'raw eda')
            
            # band-pass filter EDA
            lp_filt = signal.butter(4, 5, fs=1/dt, btype='lowpass', output='sos')
            tonic = signal.sosfiltfilt(lp_filt, eda)
  
            bp_filt = signal.butter(1, [0.03, 5], fs=1/dt, btype='bandpass', output='sos')
            phasic = signal.sosfiltfilt(bp_filt, eda)
            threshold = 0.10*np.max(phasic)
            dat = phasic
            
            
            
            

            # plt.figure(figsize=(12, 6))
            # plt.plot(t, phasic, label=f'phasic')
            # plt.plot(t, tonic-np.mean(tonic), label=f'tonic')
            # plt.axhline(y=threshold, linestyle='--', label='Threshold')
            # # plt.scatter(t[max_inds], phasic[max_inds], color='r', marker='x', label='Peaks')
            # plt.scatter(t[max_inds], tonic[max_inds]-np.mean(tonic), color='r', marker='x', label='filt Peaks')
            # # plt.scatter(t[max_inds_conv], tonic[max_inds_conv]-np.mean(tonic), color='g', marker='x', label='Conv Peaks')
            # plt.legend()
            # plt.show()
            # dat = filtered
            pass

        elif 'RESP' in signal_type:

            signal_type = 'RESP'
            unit = 'BPM'
            resp = dat

            
            filt = signal.butter(4, [1], fs=1/dt, btype='lowpass', output='sos')
            resp = signal.sosfiltfilt(filt, resp)
            peaks, _ = signal.find_peaks(resp,height=0,prominence=0.8*np.std(resp), distance=int(0.5/dt))
            peak_times = t[peaks]
            ibi = np.diff(peak_times) # inter-breath intervals in seconds

            # plt.figure(figsize=(12, 6))
            # plt.plot(t, resp, label=f'filt resp')
            # plt.scatter(t[peaks], resp[peaks], color='r', marker='x', label='Peaks')
            # plt.plot(peak_times[1:], 60./ibi, label='IBI')
            # plt.legend()
            # plt.show()
            
            dat = 60./ibi 
            t = peak_times[1:]
            pass

        preprocessed_data.append({
            "signal_type": signal_type,
            "unit": unit,
            "Timestamps": t,
            "data": dat.tolist() if hasattr(dat, "tolist") else list(dat)
        })
        
        t = daq_df['Timestamp']  # use the original timestamps for the next channel

    return preprocessed_data

def preprocess_subject_block(path, block_str, block_cfg):
    # Preprocess data
    print(f"Processing block: {block_str}")
    
    pupil_df = None
    ino_df = None
    channel_info = None
    pupil_data = None
    ino_data = None
    block_data = pd.DataFrame()

    events_file = os.path.join(path, f"{block_str}_events.csv")
    if not os.path.exists(events_file):
        print(f"Events file not found: {events_file}")
        events = None
    else:
        events = pd.read_csv(events_file)
    
    if block_cfg.get('record_pupil', False):
        pupil_file = os.path.join(path, f"{block_str}_pupil.csv")
        if not os.path.exists(pupil_file):
            print(f"Pupil data file not found: {pupil_file}")
        else:
            pupil_df = pd.read_csv(pupil_file)
            # Correct blink artifacts and filter
            pupil_data = preprocess_pupil_data(pupil_df)
            if pupil_data is not None:
                block_data['Timestamps'] = pupil_data['timestamps']
                block_data['pupilDiameter'] = pupil_data['data']

    if block_cfg.get('record_bitalino', False):
        ino_file = os.path.join(path, f"{block_str}_bitalino.csv")
        channel_file = os.path.join(path, f"{block_str}_channels.json")
        if not os.path.exists(ino_file):
            print(f"Bitalino data file not found: {ino_file}")

        else:
            ino_df = pd.read_csv(ino_file)
            if not os.path.exists(channel_file):
                print(f"Channel file not found: {channel_file}")
            else:
                channel_info = json.load(open(channel_file, 'r'))
                # preprocess ino data
                ino_df['block_path'] = path
                ino_data = preprocess_daq_data(ino_df, channel_info)

    
        
    # if pupil data is available, resample ino data to pupil timestamps
    if ino_data is not None:
        for i, channel in enumerate(ino_data):
            if channel is None or len(channel['data']) == 0:
                continue
            if 'Timestamps' in channel:
                t = channel['Timestamps']
                x = channel['data']
                if not block_data.empty:
                    # Resample to pupil timestamps (200Hz)
                    tq = block_data['Timestamps']
                else:
                    # resample to 200Hz
                    tq = np.arange(t[0], t[-1], 0.005).round(3)  # 200 Hz
                    
                block_data[channel['signal_type']] = np.interp(tq, t, x)
                block_data['Timestamps'] = tq
                
    if block_data is not None: # find indices of events in block_data timestamps
        if events is not None:
            for index, event in events.iterrows():
                # Find the closest timestamp in block_data
                event_time = event['Timestamp']
                closest_index = (block_data['Timestamps'] - event_time).abs().idxmin()
                # Add event information to block_data
                block_data.loc[closest_index, 'Event'] = event['Event']

    return block_data

def launch_peak_editor(t, ecg, peaks, block_path):
    """
    Interactive ECG peak editor.
    - Left click: select nearest peak
    - Right click: add peak at click
    - Delete/Backspace: remove selected peak
    - Left/Right arrow: move selected peak
    - Z: undo, Y: redo
    - Zoom and pan with toolbar
    """
    import os
    # Load existing corrected peaks if available
    save_path = os.path.join(block_path, "corrected_peaks.npy")
    if os.path.exists(save_path):
        corrected_peaks = np.load(save_path).tolist()
    else:
        corrected_peaks = list(peaks)
    selected_index = None
    history = []
    redo_stack = []

    # New: two subplots, ECG and HR. Do NOT share x-axis so HR stays fixed.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 7.5), sharex=False)
    plt.ion()  # enable interactive mode
    line, = ax1.plot(t, ecg, label='ECG')
    peak_plot, = ax1.plot(t[corrected_peaks], ecg[corrected_peaks], 'rx', label='Peaks')
    selected_plot, = ax1.plot([], [], 'ko', markersize=10, markerfacecolor='none', label='Selected')
    block_str = block_path.split(os.sep)[-1]
    ax1.set_title(block_str)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    # instructions for user input
    instructions = (
        "Keyboard & Mouse Controls:\n"
        "- Space / n : jump to next detected beat and center top view\n"
        "- p        : jump to previous beat and center top view\n"
        "- a        : add a peak at center of the current top view\n"
        "- Left/Right arrows : move the selected peak by one sample\n"
        "- Delete / Backspace / x : remove selected peak\n"
        "- z / y    : undo / redo\n"
        "- Tab      : reset top view to full recording\n"
        "- s or Enter : save corrected peaks and close\n"
        "- Mouse: Left click select nearest peak, Right click add peak at click\n"
    )
    plt.figtext(0, 0, instructions, wrap=True, horizontalalignment='left', fontsize=8)

    # Heart rate line on second subplot
    hr_line, = ax2.plot([], [], 'b-', label='Heart Rate (bpm)')
    ax2.set_ylabel("HR (bpm)")
    # keep HR x-axis fixed to full time series
    ax2.set_xlim([t[0], t[-1]])
    # prevent autoscaling x on HR axis
    try:
        ax2.set_autoscalex_on(False)
    except Exception:
        pass
    # marker on HR plot to indicate current selected time
    hr_marker = ax2.axvline(x=t[0], color='r', linestyle='--', linewidth=1)
    hr_marker.set_visible(False)
    plt.show(block=False)

    def update_display():
        peak_plot.set_xdata(t[corrected_peaks])
        peak_plot.set_ydata(ecg[corrected_peaks])
        if selected_index is not None and len(corrected_peaks) > 0 and selected_index <= len(corrected_peaks):
            selected_plot.set_xdata([t[corrected_peaks[selected_index]]])
            selected_plot.set_ydata([ecg[corrected_peaks[selected_index]]])
            ax1.set_xlim([max(t[0], t[corrected_peaks[selected_index]] - 2), min(t[-1], t[corrected_peaks[selected_index]] + 2)])
        else:
            selected_plot.set_xdata([])
            selected_plot.set_ydata([])
            # don't touch hr_marker here; update later
        # Update heart rate subplot
        # Update heart rate subplot only when we have consistent data
        if len(corrected_peaks) > 1:
            rr = np.diff(t[corrected_peaks])
            hr = 60 / rr
            hr_times = t[corrected_peaks][1:]
            hr_times = np.asarray(hr_times)
            hr = np.asarray(hr)
            if hr_times.size == hr.size and hr.size > 0:
                hr_line.set_xdata(hr_times)
                hr_line.set_ydata(hr)
                # autoscale only y-axis for HR while keeping x-axis fixed
                try:
                    ax2.relim()
                    ax2.autoscale_view(scalex=False, scaley=True)
                except Exception:
                    # if relim/autoscale fails, skip to avoid crashing the GUI
                    pass
            else:
                # length mismatch or empty -> clear HR line
                hr_line.set_xdata([])
                hr_line.set_ydata([])
        else:
            hr_line.set_xdata([])
            hr_line.set_ydata([])

        # Update HR marker to show currently selected peak in the timeseries
        try:
            if selected_index is not None and len(corrected_peaks) > 0 and selected_index < len(corrected_peaks):
                sel_time = float(t[corrected_peaks[selected_index]])
                # set as two points to avoid broadcasting issues
                hr_marker.set_xdata([sel_time, sel_time])
                hr_marker.set_visible(True)
            else:
                hr_marker.set_visible(False)
        except Exception:
            hr_marker.set_visible(False)
        fig.canvas.draw_idle()

    def record_state():
        history.append(list(corrected_peaks))
        while len(history) > 100:
            history.pop(0)
        redo_stack.clear()

    def onclick(event):
        nonlocal selected_index
        if event.xdata is None:
            return
        clicked_time = event.xdata
        if event.button == 1:  # Left click: select
            if len(corrected_peaks) == 0:
                return
            closest_idx = np.argmin(np.abs(t[corrected_peaks] - clicked_time))
            selected_index = closest_idx
            selected_time = t[corrected_peaks[selected_index]]
            ax1.set_xlim([selected_time - 2, selected_time + 2])
            print(f"Selected peak at {t[corrected_peaks[selected_index]]:.3f}s")
        elif event.button == 3:  # Right click: add
            new_idx = np.argmin(np.abs(t - clicked_time))
            if new_idx not in corrected_peaks:
                record_state()
                corrected_peaks.append(new_idx)
                corrected_peaks.sort()
                selected_index = corrected_peaks.index(new_idx)
                print(f"Added peak at {t[new_idx]:.3f}s")
        
        update_display()

    def onkey(event):
        nonlocal selected_index, corrected_peaks
        # normalize key (matplotlib may send ' ' or 'space')
        key = event.key if event.key is not None else ''
        key = key.lower()

        if key in ['backspace', 'delete', 'x']:
            if selected_index is not None:
                record_state()
                print(f"Removed peak at {t[corrected_peaks[selected_index]]:.3f}s")
                # remove the selected peak and pick a neighbor: prefer the next one, else previous
                corrected_peaks.pop(selected_index)
                if len(corrected_peaks) == 0:
                    selected_index = None
                else:
                    # if selection was at or beyond new length, move to last
                    if selected_index >= len(corrected_peaks):
                        selected_index = len(corrected_peaks) - 1
                    # otherwise keep same index (now points to the next peak)
                    print(f"Selected peak at {t[corrected_peaks[selected_index]]:.3f}s")

        elif key == 'left' and selected_index is not None:
            record_state()
            corrected_peaks[selected_index] = max(0, corrected_peaks[selected_index] - 1)
        elif key == 'right' and selected_index is not None:
            record_state()
            corrected_peaks[selected_index] = min(len(t) - 1, corrected_peaks[selected_index] + 1)
        elif key == 'tab':
            ax1.set_xlim([t[0], t[-1]]) # reset view to full
            selected_index = None
        elif key in [' ', 'space', 'n']:
            # Jump to the next beat (keyboard-only navigation)
            if len(corrected_peaks) == 0:
                return
            if selected_index is None:
                selected_index = 0
            else:
                selected_index = min(len(corrected_peaks) - 1, selected_index + 1)
            sel_time = t[corrected_peaks[selected_index]]
            # center top panel around selected peak with a default 4s window
            window = 4.0
            half = window / 2.0
            new_lim = [max(t[0], sel_time - half), min(t[-1], sel_time + half)]
            # adjust if near edges to keep window length
            if new_lim[1] - new_lim[0] < window:
                if new_lim[0] == t[0]:
                    new_lim[1] = min(t[-1], new_lim[0] + window)
                else:
                    new_lim[0] = max(t[0], new_lim[1] - window)
            ax1.set_xlim(new_lim)
            print(f"Selected peak at {sel_time:.3f}s")

        elif key == 'p':
            # previous peak
            if len(corrected_peaks) == 0:
                return
            if selected_index is None:
                selected_index = 0
            else:
                selected_index = max(0, selected_index - 1)
            sel_time = t[corrected_peaks[selected_index]]
            window = 4.0
            half = window / 2.0
            new_lim = [max(t[0], sel_time - half), min(t[-1], sel_time + half)]
            if new_lim[1] - new_lim[0] < window:
                if new_lim[0] == t[0]:
                    new_lim[1] = min(t[-1], new_lim[0] + window)
                else:
                    new_lim[0] = max(t[0], new_lim[1] - window)
            ax1.set_xlim(new_lim)
            print(f"Selected peak at {sel_time:.3f}s")

        elif key == 'a':
            # Add a peak at center of current view
            center_time = np.mean(ax1.get_xlim())
            new_idx = int(np.argmin(np.abs(t - center_time)))
            if new_idx not in corrected_peaks:
                record_state()
                corrected_peaks.append(new_idx)
                corrected_peaks.sort()
                selected_index = corrected_peaks.index(new_idx)
                print(f"Added peak at {t[new_idx]:.3f}s (keyboard)")

        elif key in ['s', 'enter']:
            # Save and close
            save_flag['clicked'] = True
            print('Saving corrected peaks and closing (keyboard)')
            plt.close(fig)

        elif key == 'z' and history:
            redo_stack.append(list(corrected_peaks))
            corrected_peaks = history.pop()
            selected_index = None
            print("Undo")
        elif key == 'y' and redo_stack:
            history.append(list(corrected_peaks))
            corrected_peaks = redo_stack.pop()
            selected_index = None
            print("Redo")
        update_display()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    # Save button
    save_ax = plt.axes((0.8, 0.01, 0.1, 0.05))  # x, y, width, height
    save_button = Button(save_ax, 'Save')
    save_flag = {'clicked': False}

    def on_save(event):
        save_flag['clicked'] = True
        plt.close(fig)

    save_button.on_clicked(on_save)

    update_display()
    plt.ioff()
    plt.show()
    # Save the final corrected peaks to file only if Save button was clicked
    if save_flag['clicked']:
        save_path = os.path.join(block_path, "corrected_peaks.npy")
        np.save(save_path, np.array(corrected_peaks, dtype=int))
    return np.array(corrected_peaks, dtype=int)

def main():
    parser = argparse.ArgumentParser(description='Preprocess timeseries data for paired-taVNS project')
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS/Data", help='Top-level data directory')
    parser.add_argument('--start-date', type=int, default=20250701, help='Start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=np.inf, help='End session (YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Reprocess blocks even if _tsData.csv already exists')
    parser.add_argument('--dry-run', action='store_true', help='List blocks that would be processed without writing output')
    parser.add_argument('--subject', help='Optional: only process this subject folder')
    args = parser.parse_args()

    data_dir = args.data_dir
    start_date = args.start_date
    end_date = args.end_date
    force = args.force
    dry_run = args.dry_run

    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path) or subject.startswith("test"):
            continue
        if args.subject and subject != args.subject:
            continue

        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            # skip non-directory or out-of-range sessions
            try:
                sess_int = int(session)
            except Exception:
                continue
            if not os.path.isdir(session_path) or not (start_date <= sess_int <= end_date):
                continue

            print(f"Processing {subject}/{session}...")
            for block in os.listdir(session_path):
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue

                # If output already exists and not forcing, skip this block early
                output_file = os.path.join(block_path, f"{block}_tsData.csv")
                if os.path.exists(output_file) and not force:
                    print(f"Skipping {block_path} — output exists (use --force to overwrite)")
                    continue

                try:
                    cfg_path = os.path.join(block_path, f"{block}_config.json")
                    block_cfg = json.load(open(cfg_path, 'r')) if os.path.exists(cfg_path) else {}
                    # If dry-run, just report and skip heavy processing
                    if dry_run:
                        print(f"DRY RUN: would process {block_path}")
                        continue
                    block_data = preprocess_subject_block(block_path, block, block_cfg)
                    if block_data is not None and not block_data.empty:
                        # subtract t0 from timestamps
                        t0 = block_data['Timestamps'].iloc[0]
                        block_data['Timestamps'] = np.round(pd.to_numeric(block_data['Timestamps'], errors='coerce') - t0, 3)

                        # Save block timeseries data to CSV
                        block_data.to_csv(output_file, index=False)

                        # plot data and save figures
                        plt.figure(figsize=(12, 8))
                        # Exclude 'Timestamps' and 'Event' columns for plotting
                        plot_cols = [col for col in block_data.columns if col not in ['Timestamps', 'Event', 'nSeq']]
                        num_plots = len(plot_cols)
                        for idx, col in enumerate(plot_cols, start=1):
                            plt.subplot(num_plots, 1, idx)
                            plt.plot(block_data['Timestamps'], block_data[col], label=col)
                            plt.ylabel(col)

                        plt.xlabel('Time (s)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(block_path, f"{block}_tsData.png"))
                        plt.close()

                except Exception as e:
                    tb = sys.exc_info()[2]
                    stack = traceback.extract_tb(tb)
                    func_name = stack[-1].name if stack else '<unknown>'
                    line_no = stack[-1].lineno if stack else '<unknown>'
                    print(f"Error processing {block_path}: {e} (line {line_no} in {func_name})")
                    print(traceback.format_exc())

    print(f"Data processing complete. Data directory: {data_dir}")

if __name__ == "__main__":
    main()
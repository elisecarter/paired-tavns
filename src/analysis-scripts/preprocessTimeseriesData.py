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

EVENT_ALIGNMENT_TOLERANCE_SEC = 0.025
TARGET_FS_HZ = 200
TARGET_DT_SEC = 1.0 / TARGET_FS_HZ
MAX_ALLOWED_MISSINGNESS = {
    'pupilDiameter': 0.40,
    'IBI': 0.30,
    'SD_RR': 0.40,
    'SCR': 0.35,
    'RESP': 0.35,
    'default': 0.40,
}


def _timestamp_quality_metrics(timestamps):
    arr = np.asarray(timestamps, dtype=float)
    finite = np.isfinite(arr)
    valid = arr[finite]
    if valid.size < 2:
        return {
            'n_samples': int(arr.size),
            'n_valid': int(valid.size),
            'n_invalid': int(arr.size - valid.size),
            'n_duplicates': 0,
            'n_nonmonotonic_steps': 0,
            'dt_median': np.nan,
            'dt_sd': np.nan,
            'duration_sec': np.nan,
        }

    diffs = np.diff(valid)
    return {
        'n_samples': int(arr.size),
        'n_valid': int(valid.size),
        'n_invalid': int(arr.size - valid.size),
        'n_duplicates': int(np.sum(diffs == 0)),
        'n_nonmonotonic_steps': int(np.sum(diffs <= 0)),
        'dt_median': float(np.nanmedian(diffs)),
        'dt_sd': float(np.nanstd(diffs)),
        'duration_sec': float(valid[-1] - valid[0]),
    }


def _missing_threshold_for_signal(signal_name):
    return MAX_ALLOWED_MISSINGNESS.get(signal_name, MAX_ALLOWED_MISSINGNESS['default'])


def _to_serializable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def write_integrity_report(block_path, block_str, report):
    report_path = os.path.join(block_path, f"{block_str}_integrityReport.json")
    with open(report_path, 'w') as f:
        json.dump(_to_serializable(report), f, indent=2)
    return report_path


def _build_channel_payload(signal_type, unit, timestamps, data, qa=None):
    payload = {
        "signal_type": signal_type,
        "unit": unit,
        "Timestamps": timestamps.tolist() if hasattr(timestamps, "tolist") else list(timestamps),
        "data": data.tolist() if hasattr(data, "tolist") else list(data),
    }
    if qa is not None:
        payload["qa"] = qa
    return payload

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

    qa = {
        'source': 'pupil',
        'warnings': [],
    }
    
    pupil_df = pupil_df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    timestamp_col = next((c for c in pupil_df.columns if isinstance(c, str) and 'timestamp' in c.lower()), pupil_df.columns[0])
    drop_names = {'offset', 'nseq', 'frame', 'index'}
    candidate_cols = [c for c in pupil_df.columns if c != timestamp_col and not (isinstance(c, str) and c.lower() in drop_names)]

    # Heuristic: some exports contain generic channel_* columns; pupil is expected on channel_3/channel_10.
    if len(candidate_cols) == 16 and all(isinstance(c, str) and c.lower().startswith('channel_') for c in candidate_cols):
        pupil_cols = [c for c in ['channel_3', 'channel_10'] if c in pupil_df.columns]
    else:
        pupil_cols = [c for c in candidate_cols if isinstance(c, str) and 'pupil' in c.lower()]
    if not candidate_cols:
        raise ValueError("No pupil diameter columns detected in pupil dataframe")
    if not pupil_cols:
        raise ValueError("No pupil channel matched expected names (e.g. pupil* or channel_3/channel_10)")

    # If binocular channels are present, collapse to one trace by averaging.
    if len(pupil_cols) > 1:
        pupil_data = pupil_df[pupil_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        pupil_df = pupil_df[[timestamp_col]].copy()
        pupil_df['Pupil_Diameter'] = pupil_data
    else:
        value_col = pupil_cols[0]
        pupil_df = pupil_df[[timestamp_col, value_col]].copy()
        pupil_df = pupil_df.rename(columns={value_col: 'Pupil_Diameter'})

    # Normalize numeric dtypes, then enforce monotonic unique timestamps.
    pupil_df['Timestamp'] = pd.to_numeric(pupil_df[timestamp_col], errors='coerce')
    pupil_df['Pupil_Diameter'] = pd.to_numeric(pupil_df['Pupil_Diameter'], errors='coerce')
    qa['input_timestamps'] = _timestamp_quality_metrics(pupil_df['Timestamp'].values)
    pupil_df = pupil_df.dropna(subset=['Timestamp']).drop_duplicates(subset='Timestamp', keep='first')
    pupil_df = pupil_df.sort_values(by='Timestamp').reset_index(drop=True)
    pupil_df = pupil_df.dropna(subset=['Pupil_Diameter'])
    if pupil_df.shape[0] < 2:
        raise ValueError("Pupil data has fewer than 2 valid timestamped samples after cleaning")

    pupil_diam = pupil_df['Pupil_Diameter'].values
    t = pupil_df['Timestamp'].values

    # Resample to fixed 200 Hz grid for downstream alignment with other signals.
    t_start = t[0]
    t_end = t[-1]
    dt = TARGET_DT_SEC
    uniform_t = np.arange(t_start, t_end, dt).round(3)
    if uniform_t.size < 2:
        raise ValueError("Pupil resampling grid is empty or too short")
    resampled_pupil = np.interp(uniform_t, t, pupil_diam)

    # Mark blink segments as NaN so they can be interpolated for filtering but retained as missing in output.
    blinks = detect_blinks(resampled_pupil, sampling_freq=1 / dt)
    for onset, offset in zip(blinks['blink_onset'], blinks['blink_offset']):
        onset_i = max(0, int(onset))
        offset_i = min(resampled_pupil.size, int(offset))
        resampled_pupil[onset_i:offset_i] = np.nan

    # Interpolate gaps for stable low-pass filtering, then restore blink NaNs.
    interp_pupil = pd.Series(resampled_pupil).interpolate().bfill().ffill()
    filt = signal.butter(4, 6, fs=1 / dt, btype='low', output='sos')
    pupil_filt = signal.sosfiltfilt(filt, interp_pupil)
    pupil_filt[np.isnan(resampled_pupil)] = np.nan

    # plot raw and filtered pupil data
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, pupil_diam, label='raw pupil')
    # plt.plot(uniform_t, pupil_filt, label='filtered')
    # plt.show()

    if not pupil_df.empty:
        preprocessed_data = {
            "signal_type": "PUPILDIAM",
            "unit": "mm",
            "Fs": 1/dt,
            "timestamps": uniform_t.tolist(),
            "data": pupil_filt.tolist(),
            "qa": {
                **qa,
                'resampled_timestamps': _timestamp_quality_metrics(uniform_t),
                'blink_missing_fraction': float(np.mean(np.isnan(resampled_pupil))),
                'output_missing_fraction': float(np.mean(np.isnan(pupil_filt))),
            }
        }
         
    return preprocessed_data

def preprocess_ino_data(daq_df):
    global manual_correction
    """Infer channel types from dataframe columns and filter data based on signal type."""
    if daq_df is None or daq_df.empty:
        return None

    preprocessed_data = []

    # Identify metadata columns so we only process actual signal channels.
    timestamp_col = next((c for c in daq_df.columns if c.lower() == 'timestamp'), None)
    if timestamp_col is None:
        raise ValueError("DAQ dataframe requires a 'Timestamp' column")

    meta_cols = {timestamp_col}
    offset_col = next((c for c in daq_df.columns if c.lower() == 'offset'), None)
    if offset_col:
        meta_cols.add(offset_col)
    if 'block_path' in daq_df.columns:
        meta_cols.add('block_path')

    dt = 0.001
    timestamps = pd.to_numeric(daq_df[timestamp_col], errors='coerce')
    valid_ts = timestamps.notna()
    if not valid_ts.any():
        raise ValueError("DAQ timestamps are all invalid or missing")
    dropped_ts = int((~valid_ts).sum())
    if dropped_ts > 0:
        daq_df = daq_df.loc[valid_ts].reset_index(drop=True)
        timestamps = pd.to_numeric(daq_df[timestamp_col], errors='coerce')

    # If nSeq exists, reconstruct a robust sample clock (handles counter wraparound).
    nseq_cols = [c for c in daq_df.columns if 'nseq' in c.lower()]
    if nseq_cols:
        meta_cols.update(nseq_cols)
        nSeq = pd.to_numeric(daq_df[nseq_cols[0]], errors='coerce').ffill().bfill().values
        diff = np.diff(nSeq, prepend=nSeq[0])
        wrap_mask = diff <= 0
        diff[wrap_mask] += 16
        start_candidates = timestamps.dropna()
        start_time = start_candidates.iloc[0] if not start_candidates.empty else 0.0
        timestamps = np.round(start_time + np.cumsum(diff * dt), 3)
        daq_df[timestamp_col] = timestamps
    else:
        timestamps = timestamps.values

    # Refine dt from observed spacing when possible.
    if len(timestamps) > 1:
        dt_est = np.median(np.diff(timestamps))
        if np.isfinite(dt_est) and dt_est > 0:
            dt = float(dt_est)

    base_t = pd.Series(timestamps, dtype=float)
    daq_ts_metrics = _timestamp_quality_metrics(base_t.values)

    data_cols = [c for c in daq_df.columns if c not in meta_cols and not c.lower().startswith('unnamed')]

    def _infer_type(col_name):
        name = col_name.lower()
        if 'ecg' in name:
            return 'ECG'
        if 'eda' in name or 'gsr' in name:
            return 'EDA'
        if 'resp' in name or 'breath' in name:
            return 'RESP'
        if 'nseq' in name:
            return 'NSEQ'
        return col_name

    for col in data_cols:
        channel_hint = _infer_type(col)
        if channel_hint == 'NSEQ':
            continue

        dat = pd.to_numeric(daq_df[col], errors='coerce')
        missing_before = float(dat.isna().mean())
        if dat.isnull().all():
            continue
        dat = dat.interpolate().bfill().ffill()
        t = base_t
        unit = 'a.u.'
        signal_type = channel_hint

        if channel_hint == 'ECG':
            signal_type = 'IBI'
            unit = 's'

            fs = 1 / dt
            window_size = 0.03
            ecg = pd.Series(dat, dtype=float)

            # ECG preprocessing: bandpass -> derivative^2 -> moving integration (Pan–Tompkins style).
            filt = signal.butter(4, [0.5, 30], fs=1 / dt, btype='band', output='sos')
            vFilt = signal.sosfiltfilt(filt, ecg)
            dV = np.gradient(vFilt, dt)
            dV2 = dV ** 2
            N = int(window_size / dt)
            kernel = np.ones(N) / N
            vInt = np.convolve(dV2, kernel, mode='same')

            # Peak candidates on integrated signal, then local-max correction on raw ECG.
            mpd = int(0.2 * fs)
            mph = np.mean(vInt) + 1 * np.std(vInt)
            peaks, _ = signal.find_peaks(vInt, distance=mpd, height=mph)

            corrected_peaks = []
            window_dur = 0.01
            half_window = int(window_dur / dt)
            for peak in peaks:
                if peak < half_window or peak > len(ecg) - half_window - 1:
                    continue
                window = ecg[peak - half_window: peak + half_window + 1].values
                corrected_peak = peak - half_window + np.argmax(np.array(window))
                corrected_peaks.append(corrected_peak)
            corrected_peaks = pd.Series(corrected_peaks).astype(int)

            # Optional manual correction; otherwise reuse saved corrections if present.
            block_path = daq_df['block_path'][0]
            peaks_path = os.path.join(block_path, "corrected_peaks.npy")
            if manual_correction:
                corrected_peaks = launch_peak_editor(t.values, ecg.values, corrected_peaks, block_path)
            elif os.path.exists(peaks_path):
                corrected_peaks = np.load(peaks_path).tolist()

            nn_peaks = corrected_peaks
            beatTimes = np.asarray(t.iloc[nn_peaks], dtype=float)
            if beatTimes.size < 3:
                continue
            ibi = np.diff(beatTimes)
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
            
                        
            preprocessed_data.append(
                _build_channel_payload(
                    signal_type="SD_RR",
                    unit="ms",
                    timestamps=d["Time"],
                    data=sd_rr,
                    qa={
                        'source_channel': col,
                        'source_missing_fraction': missing_before,
                        'daq_timestamps': daq_ts_metrics,
                        'dropped_invalid_timestamps': dropped_ts,
                    }
                )
            )
            # pass

        elif channel_hint == 'EDA':
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

        elif channel_hint == 'RESP':

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

        preprocessed_data.append(
            _build_channel_payload(
                signal_type=signal_type,
                unit=unit,
                timestamps=t,
                data=dat,
                qa={
                    'source_channel': col,
                    'source_missing_fraction': missing_before,
                    'daq_timestamps': daq_ts_metrics,
                    'dropped_invalid_timestamps': dropped_ts,
                }
            )
        )

    return preprocessed_data

def normalize_event_dataframe(events_df):
    if events_df is None or events_df.empty:
        return None

    events_df = events_df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    if {'Timestamp', 'Event'}.issubset(events_df.columns):
        events_df['Timestamp'] = pd.to_numeric(events_df['Timestamp'], errors='coerce')
        events_df = events_df.dropna(subset=['Timestamp'])
        return events_df[['Timestamp', 'Event']].reset_index(drop=True)

    ts_col = next((c for c in events_df.columns if isinstance(c, str) and 'timestamp' in c.lower()), None)
    if ts_col is None:
        return None

    skip_cols = {ts_col}
    skip_cols.update(c for c in events_df.columns if isinstance(c, str) and c.lower() in {'offset', 'nseq'})
    data_cols = [c for c in events_df.columns if c not in skip_cols]
    if not data_cols:
        return None

    timestamps = pd.to_numeric(events_df[ts_col], errors='coerce')
    rows = []
    for col in data_cols:
        series = events_df[col]
        if series.isnull().all():
            continue
        mask = series.notna()
        if pd.api.types.is_numeric_dtype(series):
            mask &= series != 0
        elif series.dtype == bool:
            mask &= series
        else:
            mask &= series.astype(str).str.strip().ne('')
        if not mask.any():
            continue
        active = pd.DataFrame({'Timestamp': timestamps[mask], 'Value': series[mask]})
        for _, rec in active.iterrows():
            val = rec['Value']
            if isinstance(val, str):
                val = val.strip()
            event_label = val if isinstance(val, str) and val else col
            rows.append({'Timestamp': rec['Timestamp'], 'Event': event_label})

    if not rows:
        return None

    normalized = pd.DataFrame(rows)
    normalized['Timestamp'] = pd.to_numeric(normalized['Timestamp'], errors='coerce')
    normalized = normalized.dropna(subset=['Timestamp'])
    normalized = normalized.sort_values(by='Timestamp').reset_index(drop=True)
    return normalized

def preprocess_subject_block(path, block_str, block_cfg):
    # Preprocess data
    print(f"Processing block: {block_str}")
    
    pupil_df = None
    ino_df = None
    pupil_data = None
    ino_data = None
    block_data = pd.DataFrame()
    report = {
        'block': block_str,
        'path': path,
        'policy': {
            'target_fs_hz': TARGET_FS_HZ,
            'event_alignment_tolerance_sec': EVENT_ALIGNMENT_TOLERANCE_SEC,
            'max_allowed_missingness': MAX_ALLOWED_MISSINGNESS,
        },
        'signals': {},
        'event_alignment': {
            'total_events': 0,
            'aligned_events': 0,
            'unaligned_events': 0,
            'collisions': 0,
            'residuals_sec': [],
            'unaligned_details': [],
        },
        'fail_reasons': [],
        'status': 'pass',
    }

    events_file = os.path.join(path, f"{block_str}_events.csv")
    if not os.path.exists(events_file):
        print(f"Events file not found: {events_file}")
        events = None
    else:
        try:
            events = normalize_event_dataframe(pd.read_csv(events_file))
        except pd.errors.EmptyDataError:
            print(f"Events file is empty: {events_file}")
            events = None
    
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
                if 'qa' in pupil_data:
                    report['signals']['pupilDiameter'] = pupil_data['qa']

    if block_cfg.get('record_bitalino', False):
        ino_file = os.path.join(path, f"{block_str}_bitalino.csv")
        if not os.path.exists(ino_file):
            print(f"Bitalino data file not found: {ino_file}")
        else:
            ino_df = pd.read_csv(ino_file)
            # preprocess ino data
            ino_df['block_path'] = path
            ino_data = preprocess_ino_data(ino_df)

    # if pupil data is available, resample ino data to pupil timestamps
    if ino_data is not None:
        for i, channel in enumerate(ino_data):
            if channel is None or len(channel['data']) == 0:
                continue
            if 'Timestamps' in channel:
                t = np.asarray(channel['Timestamps'], dtype=float)
                x = np.asarray(channel['data'], dtype=float)
                if t.size < 2 or x.size < 2:
                    continue

                if (not block_data.empty) and ('Timestamps' in block_data.columns):
                    # Resample to pupil timestamps (200Hz)
                    tq = np.asarray(block_data['Timestamps'], dtype=float)
                else:
                    # resample to 200Hz
                    tq = np.arange(t[0], t[-1], TARGET_DT_SEC).round(3)  # 200 Hz

                block_data[channel['signal_type']] = np.interp(tq, t, x)
                block_data['Timestamps'] = tq
                if 'qa' in channel:
                    report['signals'][channel['signal_type']] = channel['qa']
                
    if (
        block_data is not None
        and not block_data.empty
        and 'Timestamps' in block_data.columns
        and events is not None
    ):
        report['event_alignment']['total_events'] = int(events.shape[0])
        block_data['Event'] = pd.Series([None] * len(block_data), index=block_data.index, dtype='object')
        timestamps = np.asarray(block_data['Timestamps'], dtype=float)
        event_times = pd.to_numeric(events['Timestamp'], errors='coerce').to_numpy(dtype=float)
        event_labels = events['Event'].astype(str).to_numpy()

        left_bound = timestamps[0]
        right_bound = timestamps[-1]
        in_range = (event_times >= left_bound) & (event_times <= right_bound)

        for event_time, event_label in zip(event_times[~in_range], event_labels[~in_range]):
            report['event_alignment']['unaligned_details'].append({
                'timestamp': float(event_time),
                'event': str(event_label),
                'reason': 'out_of_range',
            })

        valid_times = event_times[in_range]
        valid_labels = event_labels[in_range]
        if valid_times.size:
            right_idx = np.searchsorted(timestamps, valid_times, side='left')
            right_idx = np.clip(right_idx, 0, len(timestamps) - 1)
            left_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)

            right_diff = np.abs(timestamps[right_idx] - valid_times)
            left_diff = np.abs(valid_times - timestamps[left_idx])
            use_left = left_diff <= right_diff
            nearest_idx = np.where(use_left, left_idx, right_idx)
            residuals = np.where(use_left, left_diff, right_diff)

            for event_time, event_label, idx, residual in zip(valid_times, valid_labels, nearest_idx, residuals):
                if float(residual) > EVENT_ALIGNMENT_TOLERANCE_SEC:
                    report['event_alignment']['unaligned_details'].append({
                        'timestamp': float(event_time),
                        'event': str(event_label),
                        'reason': 'outside_tolerance',
                        'nearest_residual_sec': float(residual),
                    })
                    continue

                existing_event = block_data.at[idx, 'Event']
                if pd.notna(existing_event):
                    report['event_alignment']['collisions'] += 1
                    block_data.at[idx, 'Event'] = f"{existing_event}|{event_label}"
                else:
                    block_data.at[idx, 'Event'] = event_label
                report['event_alignment']['aligned_events'] += 1
                report['event_alignment']['residuals_sec'].append(float(residual))

    report['event_alignment']['unaligned_events'] = len(report['event_alignment']['unaligned_details'])
    residuals = np.asarray(report['event_alignment']['residuals_sec'], dtype=float)
    report['event_alignment']['residual_summary'] = {
        'mean_sec': float(np.nanmean(residuals)) if residuals.size else np.nan,
        'max_sec': float(np.nanmax(residuals)) if residuals.size else np.nan,
    }

    if block_data is not None and not block_data.empty and 'Timestamps' in block_data.columns:
        report['timeline'] = _timestamp_quality_metrics(block_data['Timestamps'].values)
        for sig in [c for c in block_data.columns if c not in ['Timestamps', 'Event', 'nSeq']]:
            series = pd.to_numeric(block_data[sig], errors='coerce')
            missing_frac = float(series.isna().mean())
            threshold = _missing_threshold_for_signal(sig)
            report['signals'].setdefault(sig, {})
            report['signals'][sig]['output_missing_fraction'] = missing_frac
            report['signals'][sig]['max_allowed_missing_fraction'] = threshold
            if missing_frac > threshold:
                report['fail_reasons'].append(
                    f"{sig} missingness {missing_frac:.3f} exceeds threshold {threshold:.3f}"
                )
    else:
        report['fail_reasons'].append('No usable timeseries samples produced')

    if report['event_alignment']['unaligned_events'] > 0:
        report['fail_reasons'].append(
            f"{report['event_alignment']['unaligned_events']} event(s) were not aligned"
        )

    if report['fail_reasons']:
        report['status'] = 'fail'

    return block_data, report


def launch_block_review_gui(block_data, block_str, block_path, qa_report=None):
    """
    Interactive block-level QC review.
    Returns one of: 'accept', 'reject', 'quit'.
    Controls:
    - Buttons: Accept / Reject / Quit
    - Keyboard: a / r / q
    """
    if block_data is None or block_data.empty or 'Timestamps' not in block_data.columns:
        return 'reject'

    plot_cols = [col for col in block_data.columns if col not in ['Timestamps', 'Event', 'nSeq']]
    if len(plot_cols) == 0:
        return 'reject'

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(14, max(6, 2.2 * len(plot_cols))), sharex=True)
    if len(plot_cols) == 1:
        axes = [axes]

    t = pd.to_numeric(block_data['Timestamps'], errors='coerce')
    t = np.asarray(t, dtype=float)
    status = qa_report.get('status', 'unknown') if isinstance(qa_report, dict) else 'unknown'
    title = f"Review block: {block_str} | QA status: {status}"
    fig.suptitle(title)

    for ax, col in zip(axes, plot_cols):
        y = pd.to_numeric(block_data[col], errors='coerce')
        y = np.asarray(y, dtype=float)
        ax.plot(t, y, label=col, linewidth=1.0)
        if 'Event' in block_data.columns:
            event_times = block_data.loc[block_data['Event'].notnull(), 'Timestamps']
            for et in event_times:
                ax.axvline(x=float(et), color='r', linestyle='--', alpha=0.35)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('Time (s)')
    help_text = (
        f"Block path: {block_path}\n"
        "Review controls: [A]ccept  [R]eject  [Q]uit"
    )
    fig.text(0.01, 0.01, help_text, fontsize=9, va='bottom')

    plt.subplots_adjust(bottom=0.12, top=0.92)
    decision = {'value': None}

    accept_ax = plt.axes((0.72, 0.02, 0.08, 0.05))
    reject_ax = plt.axes((0.81, 0.02, 0.08, 0.05))
    quit_ax = plt.axes((0.90, 0.02, 0.08, 0.05))
    accept_btn = Button(accept_ax, 'Accept')
    reject_btn = Button(reject_ax, 'Reject')
    quit_btn = Button(quit_ax, 'Quit')

    def _finish(value):
        decision['value'] = value
        plt.close(fig)

    def on_accept(event):
        _finish('accept')

    def on_reject(event):
        _finish('reject')

    def on_quit(event):
        _finish('quit')

    def on_key(event):
        key = (event.key or '').lower()
        if key == 'a':
            _finish('accept')
        elif key == 'r':
            _finish('reject')
        elif key == 'q':
            _finish('quit')

    accept_btn.on_clicked(on_accept)
    reject_btn.on_clicked(on_reject)
    quit_btn.on_clicked(on_quit)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    return decision['value'] or 'reject'

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
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data", help='Top-level data directory')
    parser.add_argument('--start-date', type=int, default=20250701, help='Start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=np.inf, help='End session (YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Reprocess blocks even if _tsData.csv already exists')
    parser.add_argument('--dry-run', action='store_true', help='List blocks that would be processed without writing output')
    parser.add_argument('--subject', help='Optional: only process this subject folder')
    parser.add_argument('--review-gui', action='store_true', help='Open an interactive GUI to accept/reject each processed block before saving outputs')
    args = parser.parse_args()

    data_dir = args.data_dir
    start_date = args.start_date
    end_date = args.end_date
    force = args.force
    dry_run = args.dry_run
    review_gui = args.review_gui
    
    global manual_correction
    manual_correction = False  # set to True to enable ECG peak editor

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
                    block_data, qa_report = preprocess_subject_block(block_path, block, block_cfg)

                    if review_gui and block_data is not None and not block_data.empty:
                        review_decision = launch_block_review_gui(block_data, block, block_path, qa_report)
                        qa_report['manual_review'] = {
                            'enabled': True,
                            'decision': review_decision,
                        }
                        if review_decision == 'quit':
                            report_path = write_integrity_report(block_path, block, qa_report)
                            print(f"Wrote integrity report: {report_path}")
                            print("Review stopped by user.")
                            return
                        if review_decision == 'reject':
                            qa_report['status'] = 'fail'
                            qa_report.setdefault('fail_reasons', [])
                            qa_report['fail_reasons'].append('Rejected during manual GUI review')
                            report_path = write_integrity_report(block_path, block, qa_report)
                            print(f"Rejected block during review. Wrote integrity report: {report_path}")
                            continue

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
                            if 'Event' in block_data.columns:
                                event_times = block_data.loc[block_data['Event'].notnull(), 'Timestamps']
                                for et in event_times:
                                    plt.axvline(x=et, color='r', linestyle='--', alpha=0.5)
                            plt.ylabel(col)

                        plt.xlabel('Time (s)')
                        plt.tight_layout()
                        plt.savefig(os.path.join(block_path, f"{block}_tsData.png"))
                        plt.close()

                    report_path = write_integrity_report(block_path, block, qa_report)
                    print(f"Wrote integrity report: {report_path}")

                except Exception as e:
                    tb = sys.exc_info()[2]
                    stack = traceback.extract_tb(tb)
                    func_name = stack[-1].name if stack else '<unknown>'
                    line_no = stack[-1].lineno if stack else '<unknown>'
                    print(f"Error processing {block_path}: {e} (line {line_no} in {func_name})")
                    print(traceback.format_exc())
                    failure_report = {
                        'block': block,
                        'path': block_path,
                        'status': 'error',
                        'fail_reasons': [str(e)],
                        'error': {
                            'function': func_name,
                            'line': line_no,
                        },
                    }
                    try:
                        write_integrity_report(block_path, block, failure_report)
                    except Exception:
                        pass

    print(f"Data processing complete. Data directory: {data_dir}")

if __name__ == "__main__":
    main()
    
#Run block-by-block visual review with:
# /Users/elise/Desktop/paired-tavns/.venv/bin/python src/analysis-scripts/preprocessTimeseriesData.py --review-gui
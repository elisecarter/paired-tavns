"""
Preprocess multimodal timeseries data for paired-taVNS sessions.

Pipeline summary:
- Loads per-block pupil, BITalino, and event CSVs.
- Cleans timestamps, resamples signals to a common timeline (target: 200 Hz).
- Computes derived channels (e.g., IBI, SD_RR, SCR, RESP).
- Aligns events to nearest sample within a configurable tolerance.
- Runs automated QA checks (missingness, timestamp quality, event alignment).
- Writes block outputs (`*_tsData.csv`, optional figure) + integrity report JSON.

CLI supports date/subject filtering, dry-run, force overwrite, and manual review GUI.
"""
import os
from pathlib import Path
import pandas as pd
import json
import numpy as np
import scipy.signal as signal
import argparse
import sys
import traceback
import re
import ast
import importlib
from datetime import datetime
from typing import Any
from based_noise_blinks_detection import detect_blinks
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from pointprocess_hrv import compute_full_regression

EVENT_ALIGNMENT_TOLERANCE_SEC = 00.025
TARGET_FS_HZ = 200
TARGET_DT_SEC = 1.0 / TARGET_FS_HZ
USE_NEON_VIDEO_BLINK_DETECTION = False
LEGACY_NEON_START_DATE = 20250829
LEGACY_NEON_END_DATE = 20251028
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "real-time-blink-detection"))

PUPIL_LABS_BACKUP_ROOT = Path(r"/Volumes/WHSynology/BIOElectricsLab/Elise/pupil labs backup")
_NEON_DATETIME_DIR_CACHE = {}

# Maximum tolerated missing fraction in final output channels.
MAX_ALLOWED_MISSINGNESS = {
    'pupilDiameter': 0.50,
    'IBI': 0.50,
    'SD_RR': 0.50,
    'SCR': 0.5,
    'RESP': 0.5,
    'default': 0.50,
}

def _timestamp_quality_metrics(timestamps):
    """Return basic timestamp QA metrics (validity, monotonicity, spacing, duration)."""
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
    """Return configured missingness threshold for a signal, or default fallback."""
    return MAX_ALLOWED_MISSINGNESS.get(signal_name, MAX_ALLOWED_MISSINGNESS['default'])

def _to_serializable(value):
    """Convert NumPy/pandas-style objects to JSON-serializable Python types."""
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
    """Write per-block integrity report JSON and return output path."""
    report_path = os.path.join(block_path, f"{block_str}_integrityReport.json")
    with open(report_path, 'w') as f:
        json.dump(_to_serializable(report), f, indent=2)
    return report_path

def _build_channel_payload(signal_type, unit, timestamps, data, qa=None):
    """Build standardized channel payload for downstream merge/resampling."""
    payload = {
        "signal_type": signal_type,
        "unit": unit,
        "Timestamps": timestamps.tolist() if hasattr(timestamps, "tolist") else list(timestamps),
        "data": data.tolist() if hasattr(data, "tolist") else list(data),
    }
    if qa is not None:
        payload["qa"] = qa
    return payload

def _event_color_map(events):
    """Assign a stable tabular color to each event label."""
    labels = {
        label.strip()
        for value in events.dropna()
        for label in str(value).split('|')
        if label.strip()
    }
    colors = plt.get_cmap('tab20')
    return {label: colors(index % colors.N) for index, label in enumerate(sorted(labels))}

def _plot_event_lines(ax, timestamps, events, color_map):
    """Draw one colored vertical line for each event occurrence and label."""
    shown_labels = set()
    for timestamp, value in zip(timestamps, events):
        if pd.isna(value):
            continue
        for label in str(value).split('|'):
            label = label.strip()
            if not label:
                continue
            ax.axvline(
                x=float(timestamp),
                color=color_map[label],
                linestyle='--',
                alpha=0.55,
                label=label if label not in shown_labels else '_nolegend_',
            )
            shown_labels.add(label)

def _parse_datetime_from_text(text):
    """Extract best-effort datetime token from text using common folder/file patterns."""
    patterns = [
        (r"(\d{8})[T_\-]?(\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{8})[T_\-]?(\d{4})(?!\d)", "%Y%m%d%H%M"),
        (r"(\d{4})[-_](\d{2})[-_](\d{2})[ T_\-]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})", "%Y%m%d%H%M%S"),
        (r"(\d{4})[-_](\d{2})[-_](\d{2})[ T_\-]?(\d{2})[-_:]?(\d{2})(?![-_:]?\d)", "%Y%m%d%H%M"),
        (r"(\d{8})", "%Y%m%d"),
    ]

    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        token = "".join(match.groups())
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None

def _get_datetime_directories(root_dir):
    """Cache and return directories under root that contain parseable datetime tokens."""
    root_dir = Path(root_dir)
    cache_key = str(root_dir.resolve()) if root_dir.exists() else str(root_dir)
    if cache_key in _NEON_DATETIME_DIR_CACHE:
        return _NEON_DATETIME_DIR_CACHE[cache_key]

    candidates = []
    if not root_dir.exists():
        _NEON_DATETIME_DIR_CACHE[cache_key] = candidates
        return candidates

    for current_root, dir_names, _ in os.walk(root_dir):
        for dir_name in dir_names:
            dt = _parse_datetime_from_text(dir_name)
            if dt is None:
                continue
            full_path = Path(current_root) / dir_name
            candidates.append((dt, full_path))

    _NEON_DATETIME_DIR_CACHE[cache_key] = candidates
    return candidates

def find_closest_neon_recording_path(reference_path, backup_root=PUPIL_LABS_BACKUP_ROOT):
    """
    Find nearest same-day Neon backup directory to a block path.

    Returns:
        str | None: matched directory path or None if no acceptable match is found.
    """
    reference_path = Path(reference_path)

    # Prefer block/session folder names over the full path so we don't accidentally
    # match an earlier date token in parent directories.
    reference_dt = (
        _parse_datetime_from_text(reference_path.name)
        or _parse_datetime_from_text(reference_path.parent.name)
        or _parse_datetime_from_text(str(reference_path))
    )
    if reference_dt is None:
        return None

    candidates = _get_datetime_directories(backup_root)
    if not candidates:
        return None

    same_day_candidates = [
        (dt, path)
        for dt, path in candidates
        if dt.date() == reference_dt.date()
    ]
    if not same_day_candidates:
        print(
            f"No same-day Neon directory found for {reference_dt.date().isoformat()} under {backup_root}"
        )
        return None

    closest_dt, closest_path = min(
        same_day_candidates,
        key=lambda item: abs((item[0] - reference_dt).total_seconds()),
    )

    if abs((closest_dt - reference_dt).total_seconds()) > 60:
        print(
            f"No close Neon directory match found for {reference_dt.isoformat()} (closest: {closest_dt.isoformat()}) under {backup_root}"
        )
        return None
    
    print(f"Matched Neon recording directory: {closest_path} ({closest_dt.isoformat()})")
    return str(closest_path)

def is_legacy_neon_session(block_path):
    """Return whether a block belongs to the session range with rewritten LSL timestamps."""
    session_name = Path(block_path).parent.name
    try:
        session_date = int(session_name)
    except ValueError:
        return False
    return LEGACY_NEON_START_DATE <= session_date <= LEGACY_NEON_END_DATE

def load_fixation_cross_tag(block_path, block_str):
    """
    Load an existing fixation-cross tag JSON if present, or None otherwise.

    Annotation is done separately via `pupil-labs-posthoc/annotateFixationCross.py`
    (single block or batch) — this script never launches the tagging GUI itself.
    """
    tag_path = os.path.join(block_path, f"{block_str}_fixationCrossTimestamp.json")
    if not os.path.exists(tag_path):
        return None
    with open(tag_path, 'r') as f:
        return json.load(f)


def _load_blink_modules() -> tuple[Any, Any, str | None]:
    """Attempt optional blink-detector import; return functions and an import error."""
    try:
        helper_module = importlib.import_module("blink_detector.helper")
        detector_module = importlib.import_module("blink_detector")
        preprocess_fn = getattr(helper_module, 'preprocess_recording', None)
        pipeline_fn = getattr(detector_module, 'blink_detection_pipeline', None)
        if not callable(preprocess_fn) or not callable(pipeline_fn):
            return None, None, 'blink detector functions were not found'
        return preprocess_fn, pipeline_fn, None
    except Exception as error:
        return None, None, f'{type(error).__name__}: {error}'

def compute_neon_to_block_clock_offset(events, fixation_frame, event_label='fixation_start'):
    """Return the offset (event_timestamp - marked_timestamp) that converts Neon-clock seconds to block clock."""
    if events is None or events.empty or fixation_frame is None or 'timestamp_unix_sec' not in fixation_frame:
        return None
    event_rows = events[events['Event'] == event_label]
    if event_rows.empty:
        return None
    event_timestamp = float(event_rows.iloc[0]['Timestamp'])
    marked_timestamp = float(fixation_frame['timestamp_unix_sec'])
    return event_timestamp - marked_timestamp


def resolve_fixation_alignment_event(events, block_cfg, default_label='fixation_start'):
    """Choose the event used to align Neon data, with a StroopSquared cue fallback."""
    if events is None or events.empty:
        return default_label
    if (str(block_cfg.get('experiment', '')).lower() == 'stroopsquared'
            and not (events['Event'] == default_label).any()):
        cue_events = events[events['Event'].astype(str).str.contains('cue', case=False, na=False)]
        if not cue_events.empty:
            return str(cue_events.iloc[0]['Event'])
    return default_label


def preprocess_pupil_data(pupil_df, neon_recording_path=None, clock_offset_sec=None):
    """
    Clean, resample, blink-mask, and low-pass filter pupil diameter data.

    Notes:
    - Input channels are auto-detected (including channel_3/channel_10 heuristic).
    - Output timeline is resampled to fixed 200 Hz.
    - Blink regions are set to NaN in final output after filtering.
    - When `neon_recording_path`/`clock_offset_sec` are available, blinks are detected from the
      Neon eye-camera video (Pupil Labs' `blink_detection_pipeline`, per the notebook example)
      instead of the pupil-diameter heuristic.
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
    full_t = pupil_df['Timestamp'].values
    pupil_df = pupil_df.dropna(subset=['Pupil_Diameter'])
    if pupil_df.shape[0] < 2:
        raise ValueError("Pupil data has fewer than 2 valid timestamped samples after cleaning")

    pupil_diam = pupil_df['Pupil_Diameter'].to_numpy(dtype=float)
    t = pupil_df['Timestamp'].to_numpy(dtype=float)

    # Span the full recorded timestamp range (not just where diameter is non-NaN), so events
    # occurring during a leading/trailing dropout (e.g. blink at recording start) aren't later
    # excluded as "out_of_range" when the merged timeline gets built from these bounds.
    t_start = full_t[0]
    t_end = full_t[-1]
    dt = TARGET_DT_SEC
    uniform_t = np.arange(t_start, t_end, dt).round(3)
    if uniform_t.size < 2:
        raise ValueError("Pupil resampling grid is empty or too short")
    lsl_resampled = np.interp(uniform_t, t, pupil_diam)

    # Prefer Pupil Labs' video-based blink detector (per the notebook example) when Neon data is available.
    neon_blinks = None
    if USE_NEON_VIDEO_BLINK_DETECTION and neon_recording_path is not None and clock_offset_sec is not None:
        load_recording, run_pipeline, blink_module_error = _load_blink_modules()
        if load_recording is not None and run_pipeline is not None:
            try:
                left_images, right_images, neon_ts_ns = load_recording(neon_recording_path)
                blink_events = list(run_pipeline(left_images, right_images, neon_ts_ns))
                neon_blinks = [
                    (b.start_time / 1e9 + clock_offset_sec, b.end_time / 1e9 + clock_offset_sec)
                    for b in blink_events
                ]
                qa['blink_detection_source'] = 'pupil_labs_video'
            except Exception as e:
                qa['warnings'].append(f"Pupil Labs blink detection failed, falling back to pupil-signal heuristic: {e}")
        else:
            qa['warnings'].append(
                'Pupil Labs video blink detection unavailable; falling back to pupil-signal heuristic'
                + (f': {blink_module_error}' if blink_module_error else '')
            )

    # Mark blink segments as NaN so they can be interpolated for filtering but retained as missing in output.
    blink_onset_times = []
    blink_offset_times = []
    blink_mask = np.zeros(uniform_t.size, dtype=bool)
    if neon_blinks is not None:
        for onset_sec, offset_sec in neon_blinks:
            onset_i = max(0, min(int(np.searchsorted(uniform_t, onset_sec)), lsl_resampled.size - 1))
            offset_i = max(onset_i, min(int(np.searchsorted(uniform_t, offset_sec)), lsl_resampled.size - 1))
            blink_mask[onset_i:offset_i] = True
            blink_onset_times.append(float(uniform_t[onset_i]))
            blink_offset_times.append(float(uniform_t[offset_i]))
    else:
        qa.setdefault('blink_detection_source', 'pupil_signal_noise')
        blinks = detect_blinks(lsl_resampled, sampling_freq=1 / dt)
        for onset, offset in zip(blinks['blink_onset'], blinks['blink_offset']):
            onset_i = max(0, int(onset))
            offset_i = min(lsl_resampled.size, int(offset))
            blink_mask[onset_i:offset_i] = True
            blink_onset_times.append(float(uniform_t[onset_i]))
            blink_offset_times.append(float(uniform_t[min(offset_i, uniform_t.size - 1)]))

    lsl_masked = lsl_resampled.copy()
    lsl_masked[blink_mask] = np.nan

    # Interpolate gaps for stable low-pass filtering, then restore blink NaNs.
    filt = signal.butter(4, 6, fs=1 / dt, btype='low', output='sos')
    def _filter_with_gap_restore(values):
        filled = pd.Series(values).interpolate().bfill().ffill().to_numpy(dtype=float)
        out = signal.sosfiltfilt(filt, filled)
        out[np.isnan(values)] = np.nan
        return out

    lsl_processed = _filter_with_gap_restore(lsl_masked)
    final_diam = lsl_processed
    qa['diameter_source'] = (
        'pupil_labs_neon'
        if neon_recording_path is not None and clock_offset_sec is not None
        else 'lsl_pupil_signal'
    )
    if not USE_NEON_VIDEO_BLINK_DETECTION:
        qa['blink_detection_source'] = 'pupil_signal_noise'

    # # plot raw and filtered pupil data
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, pupil_diam, label='raw pupil')
    # plt.plot(uniform_t, final_diam, label='filtered')
    # plt.show()

    if not pupil_df.empty:
        preprocessed_data = {
            "signal_type": "PUPILDIAM",
            "unit": "mm",
            "Fs": 1/dt,
            "timestamps": uniform_t.tolist(),
            "data": final_diam.tolist(),
            "blink_onset_times": blink_onset_times,
            "blink_offset_times": blink_offset_times,
            "qa": {
                **qa,
                'resampled_timestamps': _timestamp_quality_metrics(uniform_t),
                'blink_missing_fraction': float(np.mean(np.isnan(lsl_masked))),
                'output_missing_fraction': float(np.mean(np.isnan(final_diam))),
            }
        }
         
    return preprocessed_data


def align_pupil_to_event(pupil_data, events, block_path, block_str, event_label='fixation_start', fixation_frame=None):
    """
    Mark the first occurrence of `event_label` on the pupil diameter + blink timeline
    (kept in its original block-clock timestamps), saving a QA plot and CSV.

    If `fixation_frame` (the marked Neon frame) is given, its timestamp is synchronized
    to the event's timestamp, yielding a Neon-clock -> block-clock offset.
    """
    if pupil_data is None or events is None or events.empty:
        return None

    event_rows = events[events['Event'] == event_label]
    if event_rows.empty:
        print(f"No '{event_label}' event found for {block_str}; skipping alignment")
        return None
    event_timestamp = float(event_rows.iloc[0]['Timestamp'])

    timestamps = np.asarray(pupil_data['timestamps'], dtype=float)
    blink_onsets = np.asarray(pupil_data.get('blink_onset_times', []), dtype=float)
    blink_offsets = np.asarray(pupil_data.get('blink_offset_times', []), dtype=float)

    aligned_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Pupil_Diameter': pupil_data['data'],
    })
    csv_path = os.path.join(block_path, f"{block_str}_fixationAligned.csv")
    aligned_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, pupil_data['data'], label='Pupil Diameter', linewidth=1.0)
    for onset, offset in zip(blink_onsets, blink_offsets):
        ax.axvspan(onset, offset, color='gray', alpha=0.3)
    ax.axvline(event_timestamp, color='r', linestyle='--', label=f"first '{event_label}' (t={event_timestamp:.3f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Diameter (mm)")
    ax.set_title(f"{block_str}: pupil with first '{event_label}' marked")
    ax.legend()
    plot_path = os.path.join(block_path, f"{block_str}_fixationAligned.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    result = {
        'event_label': event_label,
        'event_timestamp': event_timestamp,
        'csv_path': csv_path,
        'plot_path': plot_path,
    }

    if fixation_frame is not None and 'timestamp_unix_sec' in fixation_frame:
        marked_timestamp = float(fixation_frame['timestamp_unix_sec'])
        # Synchronize the marked frame's timestamp to the event's timestamp.
        result['marked_timestamp_unix_sec'] = marked_timestamp
        result['neon_to_block_clock_offset_sec'] = event_timestamp - marked_timestamp

    return result


def save_pupil_source_comparison(lsl_pupil_data, neon_pupil_data, block_path, block_str):
    """Save a comparison plot for filtered LSL and fixation-aligned Neon pupil traces."""
    lsl_t = np.asarray(lsl_pupil_data['timestamps'], dtype=float)
    neon_t = np.asarray(neon_pupil_data['timestamps'], dtype=float)
    lsl_values = np.asarray(lsl_pupil_data['data'], dtype=float)
    neon_values = np.asarray(neon_pupil_data['data'], dtype=float)
    t0 = min(lsl_t[0], neon_t[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lsl_t - t0, lsl_values, label='LSL pupil', linewidth=0.9, alpha=0.8)
    ax.plot(neon_t - t0, neon_values, label='Neon pupil', linewidth=0.9, alpha=0.8)
    ax.set_xlabel('Time relative to recording start (s)')
    ax.set_ylabel('Pupil Diameter (mm)')
    ax.set_title(f'{block_str}: LSL vs Neon pupil diameter')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plot_path = os.path.join(block_path, f'{block_str}_pupilSourceComparison.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def load_neon_eye_state(neon_recording_path):
    """Load Neon `eye_state` samples (pupil diameter, etc.) and their nanosecond epoch timestamps."""
    dtype_path = os.path.join(neon_recording_path, "eye_state.dtype")
    raw_path = os.path.join(neon_recording_path, "eye_state ps1.raw")
    time_path = os.path.join(neon_recording_path, "eye_state ps1.time")
    if not (os.path.exists(dtype_path) and os.path.exists(raw_path) and os.path.exists(time_path)):
        return None, None

    with open(dtype_path, 'r') as f:
        fields = ast.literal_eval(f.read())
    dtype = np.dtype(fields)

    samples = np.fromfile(raw_path, dtype=dtype)
    timestamps_ns = np.fromfile(time_path, dtype=np.int64)
    n = min(samples.size, timestamps_ns.size)
    return samples[:n], timestamps_ns[:n]


def append_neon_pupil_data(block_data, neon_recording_path, clock_offset_sec):
    """
    Append Neon `eye_state` pupil diameter (mean of both eyes) to `block_data` as
    `pupilDiameter_neon`, converting Neon-clock timestamps to block clock via
    `clock_offset_sec` and interpolating onto `block_data`'s existing timeline.
    """
    if block_data is None or block_data.empty or 'Timestamps' not in block_data.columns:
        return block_data

    samples, timestamps_ns = load_neon_eye_state(neon_recording_path)
    if samples is None or timestamps_ns is None:
        print(f"No Neon eye_state data found under {neon_recording_path}; skipping pupil append")
        return block_data

    neon_t_block = timestamps_ns.astype(np.float64) / 1e9 + clock_offset_sec
    neon_pupil = np.nanmean(
        np.vstack([samples['pupil_diameter_left_mm'], samples['pupil_diameter_right_mm']]).astype(np.float64),
        axis=0,
    )

    order = np.argsort(neon_t_block)
    neon_t_block = neon_t_block[order]
    neon_pupil = neon_pupil[order]

    tq = np.asarray(block_data['Timestamps'], dtype=float)
    block_data['pupilDiameter_neon'] = np.interp(tq, neon_t_block, neon_pupil, left=np.nan, right=np.nan)
    return block_data


def build_neon_pupil_timeline(neon_recording_path, clock_offset_sec):
    """Return Neon pupil samples on a 200 Hz block-clock timeline."""
    samples, timestamps_ns = load_neon_eye_state(neon_recording_path)
    if samples is None or timestamps_ns is None or samples.size == 0 or timestamps_ns.size == 0:
        return None, None

    required_fields = {'pupil_diameter_left_mm', 'pupil_diameter_right_mm'}
    if not required_fields.issubset(samples.dtype.names or ()):
        return None, None

    neon_t_block = timestamps_ns.astype(np.float64) / 1e9 + clock_offset_sec
    neon_pupil = np.nanmean(
        np.vstack([samples['pupil_diameter_left_mm'], samples['pupil_diameter_right_mm']]).astype(np.float64),
        axis=0,
    )
    valid = np.isfinite(neon_t_block) & np.isfinite(neon_pupil)
    if np.sum(valid) < 2:
        return None, None

    neon_t_block = neon_t_block[valid]
    neon_pupil = neon_pupil[valid]
    order = np.argsort(neon_t_block)
    neon_t_block = neon_t_block[order]
    neon_pupil = neon_pupil[order]
    unique_t, unique_indices = np.unique(neon_t_block, return_index=True)
    neon_pupil = neon_pupil[unique_indices]

    timeline = np.arange(unique_t[0], unique_t[-1], TARGET_DT_SEC).round(3)
    if timeline.size < 2:
        return None, None
    pupil = np.interp(timeline, unique_t, neon_pupil, left=np.nan, right=np.nan)
    return timeline, pupil

def preprocess_ino_data(daq_df):
    """Infer DAQ channel types, preprocess each signal, and return standardized channel payloads."""
    global manual_correction
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
                window_length=10,
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


            filt = signal.butter(4, 1, fs=1/dt, btype='lowpass', output='sos')
            resp = signal.sosfiltfilt(filt, resp)
            # peaks, _ = signal.find_peaks(resp,height=0,prominence=0.8*np.std(resp), distance=int(0.5/dt))
            # peak_times = t[peaks]
            # ibi = np.diff(peak_times) # inter-breath intervals in seconds

            # plt.figure(figsize=(12, 6))
            # plt.plot(t, resp, label=f'filt resp')
            # plt.scatter(t[peaks], resp[peaks], color='r', marker='x', label='Peaks')
            # plt.plot(peak_times[1:], 60./ibi, label='IBI')
            # plt.legend()
            # plt.show()
            
            dat = resp
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
    """
    Normalize event CSVs into canonical columns: ['Timestamp', 'Event'].

    Supports:
    - Already-normalized format with Timestamp/Event columns.
    - Wide-format boolean/numeric/string event columns keyed by timestamp.
    """
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
    """
    End-to-end preprocessing for one block directory.

    Returns:
        tuple[pd.DataFrame, dict]:
            - block_data on common timeline (signals + optional Event column)
            - integrity/QA report dictionary
    """
    # Preprocess data
    print(f"Processing block: {block_str}")
    
    pupil_df = None
    ino_df = None
    pupil_data = None
    ino_data = None
    neon_recording_path = None
    neon_clock_offset_sec = None
    neon_primary_used = False
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

    # Load events CSV and normalizes columns to ['Timestamp', 'Event']
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

    alignment_event_label = resolve_fixation_alignment_event(
        events,
        block_cfg,
        default_label=globals().get('fixation_align_event_label', 'fixation_start'),
    )
    if alignment_event_label != 'fixation_start':
        report['fixation_alignment_event'] = alignment_event_label
    
    # if pupil data is available, preprocess it and attempt to read in blinks from Neon video if available
    if block_cfg.get('record_pupil', False):
        pupil_file = os.path.join(path, f"{block_str}_pupil.csv") # raw data streamed over LSL
        if not os.path.exists(pupil_file):
            print(f"Pupil data file not found: {pupil_file}")
        else:
            pupil_df = pd.read_csv(pupil_file)
            fixation_info = load_fixation_cross_tag(path, block_str)
            neon_recording_path = find_closest_neon_recording_path(path)
            if neon_recording_path is None and fixation_info is not None:
                annotated_recording_path = fixation_info.get('recording_dir')
                if annotated_recording_path and Path(annotated_recording_path).exists():
                    neon_recording_path = annotated_recording_path
                    print(f"Using annotated Neon recording directory: {neon_recording_path}")
            neon_clock_offset_sec = None
            if neon_recording_path is None:
                print(f"No datetime directory match found under {PUPIL_LABS_BACKUP_ROOT}; using fallback blink detection")
            else:
                if fixation_info is not None:
                    report['fixation_cross'] = fixation_info
                    neon_clock_offset_sec = compute_neon_to_block_clock_offset(
                        events, fixation_info, alignment_event_label
                    )
            # Correct blink artifacts and filter
            pupil_data = preprocess_pupil_data(
                pupil_df, neon_recording_path=neon_recording_path, clock_offset_sec=neon_clock_offset_sec
            )
            if pupil_data is not None:
                lsl_pupil_data = pupil_data
                block_data['Timestamps'] = pupil_data['timestamps']
                block_data['pupilDiameter'] = pupil_data['data']
                if 'qa' in pupil_data:
                    report['signals']['pupilDiameter'] = pupil_data['qa']

                neon_timeline, neon_pupil = (None, None)
                if neon_recording_path is not None and neon_clock_offset_sec is not None:
                    neon_timeline, neon_pupil = build_neon_pupil_timeline(
                        neon_recording_path, neon_clock_offset_sec
                    )
                if neon_timeline is not None and neon_pupil is not None:
                    neon_pupil_data = preprocess_pupil_data(
                        pd.DataFrame({
                            'Timestamp': neon_timeline,
                            'pupilDiameter': neon_pupil,
                        }),
                        neon_recording_path=neon_recording_path,
                        clock_offset_sec=neon_clock_offset_sec,
                    )
                    if neon_pupil_data is None:
                        report['pupil_source'] = 'lsl_fallback'
                        report['pupil_lsl_warning'] = (
                            'Neon pupil preprocessing failed; using the LSL pupil signal.'
                        )
                    else:
                        if not is_legacy_neon_session(path):
                            comparison_path = save_pupil_source_comparison(
                                lsl_pupil_data, neon_pupil_data, path, block_str
                            )
                            report['pupil_source_comparison'] = comparison_path
                        block_data = pd.DataFrame({
                            'Timestamps': neon_pupil_data['timestamps'],
                            'pupilDiameter': neon_pupil_data['data'],
                        })
                        neon_primary_used = True
                        report['pupil_source'] = 'pupil_labs_neon'
                        report['pupil_lsl_warning'] = (
                            'Neon pupil data was filtered and blink-masked using the Neon video; '
                            'LSL data was not written to the timeseries output.'
                        )
                        report['signals']['pupilDiameter'] = neon_pupil_data['qa']
                        pupil_data = neon_pupil_data

                if neon_primary_used:
                    report['signals']['pupilDiameter']['source'] = 'pupil_labs_neon'
                else:
                    report.setdefault('pupil_source', 'lsl_fallback')
                    report.setdefault(
                        'pupil_lsl_warning',
                        'Neon pupil data was unavailable; using the LSL pupil signal.',
                    )

                if fixation_info is not None:
                    alignment = align_pupil_to_event(
                        pupil_data, events, path, block_str,
                        event_label=alignment_event_label, fixation_frame=fixation_info,
                    )
                    if alignment is not None:
                        report['fixation_alignment'] = alignment

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

    if not neon_primary_used and neon_recording_path is not None and neon_clock_offset_sec is not None:
        block_data = append_neon_pupil_data(
            block_data, neon_recording_path, neon_clock_offset_sec
        )
                
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
    event_colors = _event_color_map(block_data['Event']) if 'Event' in block_data.columns else {}
    status = qa_report.get('status', 'unknown') if isinstance(qa_report, dict) else 'unknown'
    title = f"Review block: {block_str} | QA status: {status}"
    fig.suptitle(title)

    for ax, col in zip(axes, plot_cols):
        y = pd.to_numeric(block_data[col], errors='coerce')
        y = np.asarray(y, dtype=float)
        ax.plot(t, y, label=col, linewidth=1.0)
        if 'Event' in block_data.columns:
            _plot_event_lines(ax, t, block_data['Event'], event_colors)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('Time (s)')
    if event_colors:
        handles = [
            Line2D([0], [0], color=color, linestyle='--', label=label)
            for label, color in event_colors.items()
        ]
        fig.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.90),
            fontsize=8,
            title='Events',
        )
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
    Interactive ECG peak editor with undo/redo and save support.

    Behavior:
    - Top subplot: ECG + editable peak markers.
    - Bottom subplot: derived HR trace from current peak set.
    - Saves corrected peaks to `<block_path>/corrected_peaks.npy` when confirmed.
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
        if selected_index is not None and len(corrected_peaks) > 0 and selected_index < len(corrected_peaks):
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
    """CLI entry point: iterate subjects/sessions/blocks, preprocess, review, save outputs + QA reports."""
    parser = argparse.ArgumentParser(description='Preprocess timeseries data for paired-taVNS project')
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data", help='Top-level data directory')
    parser.add_argument('--start-date', type=int, default=20250701, help='Start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=np.inf, help='End session (YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Reprocess blocks even if _tsData.csv already exists')
    parser.add_argument('--dry-run', action='store_true', help='List blocks that would be processed without writing output')
    parser.add_argument('--subject', help='Optional: only process this subject folder')
    parser.add_argument('--review-gui', action='store_true', help='Open an interactive GUI to accept/reject each processed block before saving outputs')
    parser.add_argument('--align-event-label', default='fixation_start', help='Event label whose first occurrence pupil/blink data is re-zeroed to')
    args = parser.parse_args()

    if not os.path.ismount('/Volumes/WHSynology') or not PUPIL_LABS_BACKUP_ROOT.is_dir():
        raise SystemExit(
            f"ERROR: Synology drive is not mapped or is unavailable: {PUPIL_LABS_BACKUP_ROOT}"
        )

    data_dir = args.data_dir
    start_date = args.start_date
    end_date = args.end_date
    force = args.force
    dry_run = args.dry_run
    review_gui = args.review_gui
    
    global manual_correction, fixation_align_event_label
    manual_correction = False  # set to True to enable ECG peak editor
    fixation_align_event_label = args.align_event_label

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
                # Skip already processed blocks unless --force.
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue
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
                        event_colors = _event_color_map(block_data['Event']) if 'Event' in block_data.columns else {}
                        # Exclude 'Timestamps' and 'Event' columns for plotting
                        plot_cols = [col for col in block_data.columns if col not in ['Timestamps', 'Event', 'nSeq']]
                        num_plots = len(plot_cols)
                        for idx, col in enumerate(plot_cols, start=1):
                            plt.subplot(num_plots, 1, idx)
                            ax = plt.gca()
                            ax.plot(block_data['Timestamps'], block_data[col], label=col)
                            if 'Event' in block_data.columns:
                                _plot_event_lines(
                                    ax,
                                    block_data['Timestamps'],
                                    block_data['Event'],
                                    event_colors,
                                )
                            ax.set_ylabel(col)
                            if event_colors:
                                ax.legend(loc='upper right', fontsize=8)

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
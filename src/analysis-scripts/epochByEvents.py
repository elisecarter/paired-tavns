import os
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import datetime
import argparse
import sys
import traceback

from scipy.differentiate import derivative

def epoch_by_event(block_data, event, event_df, pre_event_dur: float = 5.0, post_event_dur: float = 5.0, block_cfg=None, baseline_method='delta'):
    """Returns a tidy dataframe with trial-epoch data for the specified event."""
    # Convert timestamps to numeric seconds (handle datetime or numeric)
    if np.issubdtype(block_data['Timestamps'].dtype, np.datetime64) or hasattr(block_data['Timestamps'].dtype, 'tz'):
        times = pd.to_datetime(block_data['Timestamps']).astype('datetime64[ns]').astype('int64') / 1e9
    else:
        times = block_data['Timestamps'].to_numpy(dtype=float)

    # Compute dt (seconds) robustly
    diffs = np.diff(times)
    if len(diffs) == 0:
        return None, None
    dt = float(np.mean(diffs))
    if dt <= 0 or np.isnan(dt):
        return None, None

    pre_samples = max(1, int(np.round(pre_event_dur / dt)))
    post_samples = max(1, int(np.round(post_event_dur / dt)))

    timeseries_data = []
    feature_data = []

    # iterate events but map event index labels to positional indices in block_data
    for i, (event_idx, event_row) in enumerate(event_df.iterrows()):
        try:
            pos = block_data.index.get_loc(event_idx)
        except Exception:
            # fallback to integer position lookup
            pos_list = block_data.index.get_indexer([event_idx])
            if len(pos_list) == 0 or pos_list[0] == -1:
                continue
            pos = int(pos_list[0])

        start_ind = pos - pre_samples
        end_ind = pos + post_samples
        if start_ind < 0 or end_ind > (len(block_data) - 1):
            continue

        rel_time = times[start_ind:end_ind] - times[pos]

        # Compute response time (RT) as latency to next response event after this event onset
        rt = np.nan
        try:
            if 'Event' in block_data.columns:
                # Look ahead for the next response event; prefer 'correct' if present
                next_events = block_data['Event'].iloc[pos+1:]
                resp_mask = next_events.astype(str).str.contains('response', case=False, na=False)
                if resp_mask.any():
                    resp_indices = next_events[resp_mask].index
                    # prefer first 'correct' response among response events
                    next_resp_series = block_data.loc[resp_indices, 'Event'].astype(str)
                    correct_mask = next_resp_series.str.contains('correct', case=False, na=False)
                    if correct_mask.any():
                        first_idx = next_resp_series[correct_mask].index[0]
                    else:
                        first_idx = resp_indices[0]
                    resp_pos = block_data.index.get_loc(first_idx)
                    rt = float(times[resp_pos] - times[pos])
        except Exception:
            rt = np.nan

        for sig in block_data.columns.difference(['Timestamps', 'Event', 'nSeq']):
            values = block_data[sig].iloc[start_ind:end_ind].to_numpy(dtype=float)
            
            # outlier_threshold = 3 * np.nanstd(values)
            # outliers = np.abs(values - np.nanmean(values)) > outlier_threshold
            # values[outliers] = np.nan  # mark outliers as NaN
            # if sig == 'pupilDiameter':
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(rel_time, values - np.nanmean(values), label='Original')
            #     plt.axhline(y=outlier_threshold, color='r', linestyle='--', label='Outlier Threshold')
            #     plt.axhline(y=-outlier_threshold, color='r', linestyle='--')
            #     plt.title(f'percent bad data: {np.sum(np.isnan(values)) / len(values) * 100:.2f}%')
            #     plt.legend()
            #     plt.show()
            
            # determine percent interpolated
            if np.sum(np.isnan(values)) / len(values) > 0.50:
                # skip trials with >30% bad/missing data
                print(f"Skipping trial {i} for event {event} due to excessive missing data ({np.sum(np.isnan(values)) / len(values) * 100:.2f}%)")
                continue

            # interpolate missing data linearly
            nans = np.isnan(values)
            if np.any(nans):
                values[nans] = np.interp(rel_time[nans], rel_time[~nans], values[~nans])
             
    
            
            

            # baseline window derived from pre_event_dur
            BL_mask = (rel_time >= -pre_event_dur) & (rel_time < 0)
            BL_vals = values[BL_mask]
            if BL_vals.size == 0:
                BL_mean = 0.0
                BL_std = 1.0
                baseline_flag = False
            else:
                BL_mean = np.nanmean(BL_vals)
                BL_std = np.nanstd(BL_vals)
                baseline_flag = True
                if np.isnan(BL_std) or BL_std == 0:
                    # avoid divide-by-zero; mark invalid baseline
                    BL_std = 1.0
                    baseline_flag = False

            # apply selected baseline correction method
            if baseline_method == 'zscore':
                z_values = (values - BL_mean) / BL_std
            elif baseline_method == 'delta':
                z_values = values - BL_mean
            elif baseline_method == 'percent_change':
                if BL_mean != 0:
                    z_values = ((values - BL_mean) / BL_mean) * 100.0
                else:
                    z_values = values * 0.0  # avoid divide-by-zero
            elif baseline_method == 'none':
                z_values = values # no baseline correction

            # Extract features on post-event window (extract_features expects numpy arrays)
            features = extract_features(z_values, rel_time, sig)
            
            
            
            # if PLRT, caluclate additional features
            if 'PLRT' in (block_cfg.get('experiment', '') if block_cfg else '') and sig == 'pupilDiameter':
                plrt_features = extract_plrt_features(z_values, rel_time)
                if isinstance(features, dict):
                    features.update(plrt_features)
                else:
                    features = plrt_features
            

            # build long-form timeseries rows (one row per sample)
            n_samples = len(rel_time)
            if n_samples != len(z_values):
                continue

            # canonical id field from block_cfg
            subj_id = None
            if block_cfg:
                subj_id = block_cfg.get('participant_ID', block_cfg.get('ID'))

            ts_df = pd.DataFrame({
                'id': subj_id,
                'order': block_cfg.get('order') if block_cfg else None,
                'datetime': block_cfg.get('datetime') if block_cfg else None,
                'condition': block_cfg.get('condition') if block_cfg else None,
                'experiment': block_cfg.get('experiment') if block_cfg else None,
                'block': block_cfg.get('block_no') if block_cfg else None,
                'trial': int(event_df['trial'].iloc[i]) if 'trial' in event_df.columns else None,
                'time': np.round(rel_time, 3),
                'signal_type': sig,
                'event': str(event).strip(),
                'value': np.round(z_values, 3),
                'rt': np.round(rt, 3)
            })

            # explode arrays into long-form rows
            # Use explode for both columns and ensure numeric scalars (not arrays/objects) for plotting
            ts_long = ts_df.explode(['time', 'value']).reset_index(drop=True)
            # coerce types to numeric to avoid object/array cells that matplotlib may interpret oddly
            ts_long['time'] = pd.to_numeric(ts_long['time'], errors='coerce')
            ts_long['value'] = pd.to_numeric(ts_long['value'], errors='coerce')
            # ensure RT numeric
            if 'rt' in ts_long.columns:
                ts_long['rt'] = pd.to_numeric(ts_long['rt'], errors='coerce')
            timeseries_data.append(ts_long)

            # feature dict
            feature_dict = {
                'id': subj_id,
                'order': block_cfg.get('order') if block_cfg else None,
                'datetime': block_cfg.get('datetime') if block_cfg else None,
                'condition': block_cfg.get('condition') if block_cfg else None,
                'experiment': block_cfg.get('experiment') if block_cfg else None,
                'block': block_cfg.get('block_no') if block_cfg else None,
                'trial': int(event_df['trial'].iloc[i]) if 'trial' in event_df.columns else None,
                'event': str(event).strip(),
                'signal_type': sig,
                'baseline_valid': baseline_flag,
                'baseline_mean': float(BL_mean),
                'baseline_std': float(BL_std)
            }
            if isinstance(features, dict):
                feature_dict.update(features)
            feature_data.append(pd.DataFrame([feature_dict]))

    # Safely concatenate only when we have collected data; otherwise return (None, None)
    timeseries_df = pd.concat(timeseries_data, ignore_index=True) if timeseries_data else None
    feature_df = pd.concat(feature_data, ignore_index=True) if feature_data else None

    if timeseries_df is None and feature_df is None:
        return None, None

    return timeseries_df, feature_df

def extract_plrt_features(trial_data, time):
    """Extracts PLRT-specific features from pupil diameter trial data."""
    features = {}
    if trial_data is None:
        return features
    
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.0
    velocity = np.gradient(trial_data, dt) if dt > 0 else np.array([np.nan]*len(trial_data))
    
    # maximum constriction velocity: window 0-1s
    constriction_mask = (time >= 0) & (time <= 1)
    const_t = time[constriction_mask]
    mcv = np.min(velocity[constriction_mask]) if dt > 0 else np.nan # maximum constriction velocity
    mcv_latency = const_t[np.argmin(velocity[constriction_mask])] if dt > 0 else np.nan
    
    # maximum dilation velocity: window relative to mcv
    dilation_mask = (time > 0.5) & (time <= mcv_latency+1.0)
    dil_t = time[dilation_mask]
    mdv = np.max(velocity[dilation_mask]) if dt > 0 else np.nan 
    mdv_latency = dil_t[np.argmax(velocity[dilation_mask])] if dt > 0 else np.nan
    
    # constriction amplitude/latency (between mcv and mdv)
    if dt > 0 and not np.isnan(mcv_latency) and not np.isnan(mdv_latency):
        if mcv_latency < mdv_latency:
            ca_mask = (time >= mcv_latency) & (time <= mdv_latency)
            if np.any(ca_mask):
                ca_data = trial_data[ca_mask]
                const_amplitude = ca_data.min() - trial_data[trial_data == trial_data[np.argmin(velocity)]][0]
                const_latency = time[ca_mask][ca_data.argmin()] if len(ca_data) > 0 else np.nan
            else:
                const_amplitude = np.nan
                const_latency = np.nan
        else:
            const_amplitude = np.nan
            const_latency = np.nan
    features.update({
        'max_constriction_velocity': mcv,
        'max_dilation_velocity': mdv,
        'mcv_latency': mcv_latency,
        'mdv_latency': mdv_latency,
        'const_amplitude': const_amplitude,
        'const_latency': const_latency,
    })
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(time, trial_data, label='pupilDiameter')
    # plt.axvline(x=mcv_latency, color='r', linestyle='--', label='Max Constriction Velocity')
    # plt.axvline(x=mdv_latency, color='g', linestyle='--', label='Max Dilation Velocity')
    # plt.axvline(x=const_latency, color='b', linestyle='--', label='Constriction Latency')
    # plt.legend()
    # plt.show()

    return features

def extract_features(trial_data, time, signal, thresh=0):
    """Extracts features from the trial timeseries."""
    if trial_data is None:
        return None
    

    post_event_mask = (time >= 0)
    trial_data = trial_data[post_event_mask]
    time = time[post_event_mask]
    
    # if signal == 'pupilDiameter' or signal == 'EDA': "Detect event-related pupil response"
    max_value = trial_data.max()
    min_value = trial_data.min()
    max_time = time[trial_data.argmax()]
    min_time = time[trial_data.argmin()]
    
    
    # use trapz for numerical integration; pass x=time to be safe
    try:
        auc = np.trapezoid(trial_data, x=time)
        abs_auc = np.trapezoid(np.abs(trial_data), x=time)
    except Exception:
        auc = np.nan
        abs_auc = np.nan

    features = {
        'mean': trial_data.mean(),
        'std': trial_data.std(),
        'median': np.median(trial_data),
        'iqr': np.subtract(*np.percentile(trial_data, [75, 25])),
        'min': min_value,
        'max': max_value,
        'time_to_max': max_time,
        'time_to_min': min_time,
        'auc': auc,
        'abs_auc': abs_auc,
    }
        
    
        
    if signal == 'SCR':
        signs = np.sign(trial_data - thresh) # threshold at zero
        zcs = np.concatenate(([0], np.diff(signs)))  # prepend 0 to match length

        # find positive zero crossings
        pos_crossings = np.where(zcs > 0)
        neg_crossings = np.where(zcs < 0)

        max_inds = []
        num_peaks = 0
        for p in pos_crossings[0]:
            # if p > neg_crossings[0][-1]:
            #     continue
            # Find next largest index in neg_crossings
            if any(neg_crossings[0] > p):
                n = neg_crossings[0][neg_crossings[0] > p][0]
            else:
                n = len(trial_data)
            # Find max_index in phasic from p -> n
            max_inds.append(np.argmax(trial_data[p:n]) + p)
            num_peaks += 1
        # features['num_peaks'] = num_peaks
        
        
        # plt.figure(figsize=(10, 5))
        # sns.lineplot(x=time, y=trial_data, label=signal)
        # plt.axhline(y=max_value, color='r', linestyle='--', label='Max Peak')
        # plt.axhline(y=min_value, color='g', linestyle='--', label='Min Trough')
        # plt.axvline(x=peak_time, color='r', linestyle=':', label='Peak Time')
        # plt.axvline(x=trough_time, color='g', linestyle=':', label='Trough Time')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Pupil Diameter (z)')
        # plt.legend()
        # plt.show()

    # # Extract features
    # features = {
    #     'mean': trial_data.mean(),
    #     'std': trial_data.std(),
    #     'min': trial_data.min(),
    #     'max': trial_data.max(),
    #     'median': trial_data.median(),
    #     '25%': trial_data.quantile(0.25),
    #     '75%': trial_data.quantile(0.75),
    # }
    

    return features

def compute_timeseries_metrics(g: pd.DataFrame) -> dict:
    """Compute metrics on an average subject/condition trace (columns: time, val)."""
    try:
        t = g['time'].to_numpy(dtype=float)
        v = g['val'].to_numpy(dtype=float)
        if t.size == 0 or v.size == 0:
            return {}
        # ensure sorted by time
        order = np.argsort(t)
        t = t[order]
        v = v[order]
        # basic stats
        v_mean = float(np.nanmean(v))
        v_std = float(np.nanstd(v))
        v_median = float(np.nanmedian(v))
        v_iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        v_min = float(np.nanmin(v))
        v_max = float(np.nanmax(v))
        # times to extrema
        max_idx = int(np.nanargmax(v))
        min_idx = int(np.nanargmin(v))
        t_to_max = float(t[max_idx])
        t_to_min = float(t[min_idx])
        # area under curve
        try:
            auc = float(np.trapz(v, x=t))
            abs_auc = float(np.trapz(np.abs(v), x=t))
        except Exception:
            auc = np.nan
            abs_auc = np.nan
        return {
            'mean': v_mean,
            'std': v_std,
            'median': v_median,
            'iqr': v_iqr,
            'min': v_min,
            'max': v_max,
            'time_to_max': t_to_max,
            'time_to_min': t_to_min,
            'auc': auc,
            'abs_auc': abs_auc,
        }
    except Exception:
        return {}

# ---------------------------
# Main Pipeline
# ---------------------------

def main():
    today = datetime.datetime.today().strftime('%Y%m%d')
    
    parser = argparse.ArgumentParser(description='Preprocess timeseries data for paired-taVNS project')
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data", help='Top-level data directory')
    parser.add_argument('--output-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/analyzed-data", help='Output directory for processed data')
    parser.add_argument('--start-date', type=int, default=20250701, help='Start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=np.inf, help='End session (YYYYMMDD)')
    parser.add_argument('--dry-run', action='store_true', help='List blocks that would be processed without writing output')
    parser.add_argument('--subject', help='Optional: only process this subject folder')
    parser.add_argument('--baseline-correct-method', choices=['zscore', 'delta', 'percent_change', 'none'], default='percent_change', help='Baseline correction method to apply to epochs')
    args = parser.parse_args()

    data_dir = args.data_dir
    date_dir = os.path.join(args.output_dir, today)
    output_dir = os.path.join(date_dir, args.baseline_correct_method)
    start_date = args.start_date
    end_date = 20260201
    dry_run = args.dry_run
    os.makedirs(output_dir, exist_ok=True)

    # master accumulator for all block timeseries to compute experiment-level averages
    master_timeseries = []
    # condition colors (hex) — used for plotting
    condition_colors = {
        'sham': '#6dc8bf',
        'taVNS': '#f15a22'
    }
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if not os.path.isdir(subject_path) or subject.startswith("test") or subject.startswith('analyzed'):
            continue
        if args.subject and subject != args.subject:
            continue

        print(f"Processing {subject}...")
        
        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            if not os.path.isdir(session_path) or not (start_date <= int(session) <= end_date):
                continue
            if (int(session) > 20250919 and int(session) < 20251118) or (int(session) == 20250829) or (int(session) == 20251121):
                # skip buggy sessions
                continue
            
            for block in os.listdir(session_path):
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue
                print(f"  Processing block {block}...")
                feature_data = []
                # accumulate timeseries for this block (list of long-form dfs)
                block_timeseries = []
                # check for stroop table in block directory
                if os.path.exists(os.path.join(block_path, f"{block}_stroopTrials.csv")):
                    # load stroop trials
                    stroop_trials = pd.read_csv(os.path.join(block_path, f"{block}_stroopTrials.csv"))
                    # rename trial column to 'trial' if it exists
                    if 'trial_number' in stroop_trials.columns:
                        stroop_trials.rename(columns={'trial_number': 'trial'}, inplace=True)
                else:
                    stroop_trials = None
                
                try:
                    block_cfg = json.load(open(os.path.join(block_path, f"{block}_config.json"), 'r'))
                    ts_data = pd.read_csv(os.path.join(block_path, f"{block}_tsData.csv"))
                    if ts_data.empty:
                        print(f"No time series data found in {block_path}")
                        continue
                    
                    experiment_type = block_cfg.get('experiment', '')
                    if 'PLRT' in experiment_type:
                        pre, post = 0.2, 3
                    elif 'SCWT' in experiment_type:
                        pre, post = 1, 2
                    elif 'StroopSquared' in experiment_type:
                        pre, post = 1, 2
                    else:
                        continue

                    df_events = pd.DataFrame({'event': ts_data['Event'].dropna()})
                    # Only filter events for SCWT or StroopSquared experiments
                    if experiment_type in ['SCWT', 'StroopSquared', 'PLRT']:
                        # remove cue/stimulus trials that are not proceeded by correct response
                        # Filter events to only those containing "flash_start", "cue", or "response"
                        df_events = df_events[df_events['event'].str.contains('flash_start|cue|response', case=False, na=False)]

                        # iterate by positional index to safely get next-event responses
                        for pos in range(len(df_events)):
                            idx = df_events.index[pos]
                            event = str(df_events['event'].iloc[pos])
                            if 'cue' in event.lower():
                                if pos + 1 < len(df_events):
                                    next_event = str(df_events['event'].iloc[pos + 1])
                                    response = next_event.split('_')[-1]  # Get the last part after underscore
                                    df_events.at[idx, 'event'] = f"{event}_{response}"
                                else: # delete event if no next event
                                    df_events.at[idx, 'event'] = None
                            # if 'stim' in event.lower():
                            #     if "_" in event: # strip condition from stimulus event
                            #         event_prefix = event.split('_')[0]
                            #         df_events.at[idx, 'event'] = event_prefix


                        response_mask = df_events['event'].str.contains('response_', case=False, na=False)
                        df_events = df_events.loc[~response_mask]
                        # Default trial numbering
                        df_events['trial'] = np.arange(1, len(df_events) + 1, dtype=int)
                        # df_events.loc[stim_cue_mask, 'trial'] = np.arange(1, stim_cue_mask.sum() + 1, dtype=int)
                        # df_events['trial'] = df_events['trial'].ffill()
                        # # Check if the next event is a "correct" response
                        # correct_next_mask = stim_cue_mask & df_events['event'].shift(-1).str.contains('correct', case=False, na=False)
                        # df_events = df_events[correct_next_mask | df_events['event'].str.contains('response', case=False, na=False)]
                        # add trial number to df_events

                        
                    # Only use events present in df_events for epoching
                    marker = df_events['event'].unique().tolist()

                    # Collect both timeseries (long-form) and feature-level data
                    for event in marker:
                        event_df = df_events[df_events['event'] == event]
                        trial_data = epoch_by_event(ts_data, event, event_df, pre_event_dur=pre, post_event_dur=post, block_cfg=block_cfg, baseline_method=args.baseline_correct_method)
                        if trial_data is not None:
                            ts, feat = trial_data
                            # accumulate timeseries for block-level averaging
                            if ts is not None and not ts.empty:
                                block_timeseries.append(ts)
                            # feature rows
                            if feat is not None and not feat.empty:
                                feature_data.append(feat)

                    # If we collected any feature rows, concatenate and save
                    if feature_data:
                        feature_df = pd.concat(feature_data, ignore_index=True)
                        if stroop_trials is not None:
                            # Merge with stroop trials if available
                            feature_df = feature_df.merge(stroop_trials, on=['trial'], how='left')
                        # Save raw feature rows per experiment
                        exp_name = str(block_cfg.get('experiment', 'unknown')).strip() if block_cfg else 'unknown'
                        exp_feat_dir = os.path.join(output_dir, 'features_by_experiment')
                        os.makedirs(exp_feat_dir, exist_ok=True)
                        csv_file = os.path.join(exp_feat_dir, f"features-table_{exp_name}.csv")
                        feature_df.to_csv(csv_file, index=False, mode="a", header=not os.path.exists(csv_file))

                        # # Compute condition averages per (id, block, condition)
                        # # choose numeric feature columns to average
                        # numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
                        # # exclude 'trial' from aggregated means (not a signal feature)
                        # if 'trial' in numeric_cols:
                        #     numeric_cols.remove('trial')

                        # if len(numeric_cols) > 0:
                        #     agg_block = feature_df.groupby(['id', 'block', 'condition'])[numeric_cols].mean().reset_index()
                        #     agg_subject = feature_df.groupby(['id', 'condition'])[numeric_cols].mean().reset_index()
                        #     agg_group = feature_df.groupby(['condition'])[numeric_cols].mean().reset_index()
                            
                            


                    # -------------------------
                    # Block-level timeseries averaging and plotting
                    # -------------------------
                    if block_timeseries:
                        block_ts_df = pd.concat(block_timeseries, ignore_index=True)
                        # ensure numeric types
                        block_ts_df['time'] = pd.to_numeric(block_ts_df['time'], errors='coerce')
                        block_ts_df['value'] = pd.to_numeric(block_ts_df['value'], errors='coerce')

                        # # save a CSV per block with the raw concatenated timeseries
                        # block_out_dir = os.path.join(output_dir, 'timeseries_by_block')
                        # os.makedirs(block_out_dir, exist_ok=True)
                        # block_csv = os.path.join(block_out_dir, f"{block}_timeseries_raw.csv")
                        # block_ts_df.to_csv(block_csv, index=False)

                        # append block-level timeseries to master list for experiment-level aggregation
                        master_timeseries.append(block_ts_df)

                except Exception as e:
                    tb = sys.exc_info()[2]
                    stack = traceback.extract_tb(tb)
                    func_name = stack[-1].name if stack else '<unknown>'
                    line_no = stack[-1].lineno if stack else '<unknown>'
                    print(f"Error processing {block_path}: {e} (line {line_no} in {func_name})")
                    print(traceback.format_exc())
        
    # ----------------------------
    # After main run: compute experiment-level overlays (mirror plotTimeSeries.R)
    # ----------------------------
    try:
        if master_timeseries:
            all_ts = pd.concat(master_timeseries, ignore_index=True)
            # coerce types
            all_ts['time'] = pd.to_numeric(all_ts['time'], errors='coerce')
            all_ts['value'] = pd.to_numeric(all_ts['value'], errors='coerce')

            # remove practice condition if present
            all_ts = all_ts[all_ts['condition'] != 'practice'] if 'condition' in all_ts.columns else all_ts

            # loop over experiments, events, signals similar to R script
            for exp in all_ts['experiment'].dropna().unique():
                exp_df = all_ts[all_ts['experiment'] == exp]
                exp_out_dir = os.path.join(output_dir, f"timeseries_plots_{exp}")
                os.makedirs(exp_out_dir, exist_ok=True)

                for ev in exp_df['event'].dropna().unique():
                    ev_out_dir = os.path.join(exp_out_dir, f"{ev}")
                    os.makedirs(ev_out_dir, exist_ok=True)
                    ev_df = exp_df[exp_df['event'] == ev]
                    for sig in ev_df['signal_type'].dropna().unique():
                        sig_df = ev_df[ev_df['signal_type'] == sig]


                        
                        block_mean = sig_df.groupby(['experiment', 'signal_type', 'event', 'id', 'block', 'condition', 'time'])['value'].mean().reset_index().rename(columns={'value': 'val'})
                        

                        # per-subject mean trace (id x condition x time)
                        subject_df = sig_df.groupby(['experiment', 'signal_type', 'event', 'id', 'condition', 'time'])['value']
                        subject_mean = subject_df.mean().reset_index().rename(columns={'value': 'val'})

                        # Collect metrics from average timeseries per subject/condition
                        metrics_rows = []
                        if not subject_mean.empty:
                            for (sid, cond), sgroup in subject_mean.groupby(['id', 'condition']):
                                m = compute_timeseries_metrics(sgroup[['time', 'val']])
                                for k, v in m.items():
                                    metrics_rows.append({
                                        'expt': exp,
                                        'event': ev,
                                        'signal': sig,
                                        'id': sid,
                                        'condition': cond,
                                        'metric': k,
                                        'value': v,
                                    })

                        # condition mean and sem across subjects
                        condition_mean = subject_mean.groupby(['experiment', 'signal_type', 'event', 'condition', 'time']).agg(
                            n_subjects=('id', 'nunique'),
                            mean=('val', 'mean'),
                            sem=('val', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                        ).reset_index()

                        
                    
                        # plotting condition mean with subject traces overlay
                        plt.figure(figsize=(5.5,4.25))
                        plt.axvline(0, color='k', linestyle='--')
                        # plot ribbons per condition
                        for cond, cond_df in condition_mean.groupby('condition'):
                            # choose color for this condition, fallback to default
                            col = condition_colors.get(str(cond), None)
                            if col is not None:
                                plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
                                plt.plot(cond_df['time'], cond_df['mean'], color=col, label=f"{cond} (mean)")
                            else:
                                plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
                                plt.plot(cond_df['time'], cond_df['mean'], label=f"{cond} (mean)")

                        # For SCWT pupil response, plot mean RT per condition as vertical lines
                        if str(exp) == 'SCWT' and str(sig) == 'pupilDiameter' and 'rt' in sig_df.columns:
                            try:
                                rt_trials = sig_df[['condition', 'id', 'datetime', 'trial', 'rt']].dropna(subset=['rt'])
                                if not rt_trials.empty:
                                    rt_trials = rt_trials.drop_duplicates(subset=['condition', 'id', 'datetime', 'trial'])
                                    rt_means = rt_trials.groupby('condition')['rt'].mean()
                                    for cond, rt_val in rt_means.items():
                                        col = condition_colors.get(str(cond), None)
                                        plt.axvline(float(rt_val), color=col if col is not None else 'gray', linestyle=':', linewidth=1.5, label=f"{cond} mean RT")
                            except Exception:
                                pass

                        # overlay subject traces
                        if not subject_mean.empty:
                            for (sid, cond), sgroup in subject_mean.groupby(['id', 'condition']):
                                col = condition_colors.get(str(cond), None)
                                plt.plot(sgroup['time'], sgroup['val'], alpha=0.2, linewidth=0.6, label=None, color=col)

                        plt.xlabel('Time (s)')
                        plt.ylabel(args.baseline_correct_method)
                        plt.title(f"{exp} : {ev} : {sig}")
                        
                        plt.legend()
                        plt.tight_layout()
                        plotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}.svg")
                        plt.savefig(plotfile, dpi=300)
                        plt.close()

                        # Save metrics table for this experiment/event/signal
                        if metrics_rows:
                            metrics_df = pd.DataFrame(metrics_rows)
                            metrics_dir = os.path.join(output_dir, 'metrics_by_timeseries')
                            os.makedirs(metrics_dir, exist_ok=True)
                            metrics_csv = os.path.join(metrics_dir, 'timeseries_metrics.csv')
                            metrics_df.to_csv(metrics_csv, index=False, mode='a', header=not os.path.exists(metrics_csv))

                        # ---------------------------------------
                        # Contrast: taVNS - sham (by subject, then mean+SEM)
                        # ---------------------------------------
                        try:
                            # Build wide condition table per subject/time
                            wide = subject_mean.pivot_table(index=['id', 'time'], columns='condition', values='val')
                            # Ensure both conditions are present
                            if {'taVNS', 'sham'}.issubset(set(wide.columns)):
                                wide = wide.reset_index()
                                wide['contrast'] = wide['taVNS'] - wide['sham']
                                contrast_subject = wide[['id', 'time', 'contrast']]

                                # Group mean and SEM across subjects
                                contrast_group = contrast_subject.groupby('time').agg(
                                    n_subjects=('id', 'nunique'),
                                    mean=('contrast', 'mean'),
                                    sem=('contrast', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                                ).reset_index()

                                # Plot contrast ribbon + mean, with faint subject lines
                                plt.figure(figsize=(8,5))
                                plt.fill_between(contrast_group['time'], contrast_group['mean'] - contrast_group['sem'], contrast_group['mean'] + contrast_group['sem'], linewidth=0, color='#333333', alpha=0.3, label='contrast (sem)')
                                plt.plot(contrast_group['time'], contrast_group['mean'], color='#333333', label='contrast (mean)')

                                for sid, sgroup in contrast_subject.groupby('id'):
                                    plt.plot(sgroup['time'], sgroup['contrast'], alpha=0.2, linewidth=0.6, label=None, color='#333333')

                                plt.xlabel('Time (s)')
                                plt.ylabel(args.baseline_correct_method)
                                plt.title(f"{exp} : {ev} : {sig} : taVNS - sham")
                                plt.axvline(0, color='k', linestyle='--')
                                plt.legend()
                                plt.tight_layout()
                                cplotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}_contrast.svg")
                                plt.savefig(cplotfile, dpi=300)
                                plt.close()
                        except Exception:
                            # Keep overlays robust; skip contrast if any issue
                            pass

                        # ---------------------------------------
                        # First derivative plots (pupilDiameter only)
                        # ---------------------------------------
                        if str(sig) == 'pupilDiameter' and not subject_mean.empty:
                            def _compute_deriv(g: pd.DataFrame) -> pd.DataFrame:
                                g = g.sort_values('time')
                                t = g['time'].to_numpy()
                                v = g['val'].to_numpy()
                                if len(t) < 2:
                                    g['dval_dt'] = np.nan
                                    return g
                                try:
                                    deriv = np.gradient(v, t)
                                except Exception:
                                    # Fallback: forward differences with NaN at start
                                    dv = np.diff(v)
                                    dt = np.diff(t)
                                    deriv = np.concatenate([[np.nan], np.divide(dv, dt, out=np.full_like(dv, np.nan, dtype=float), where=dt!=0)])
                                g['dval_dt'] = deriv
                                return g

                            # Per-subject derivative traces (id x condition x time)
                            subj_deriv = subject_mean.groupby(['id', 'condition'], group_keys=False).apply(_compute_deriv)

                            # Condition mean and SEM over subjects
                            cond_deriv = subj_deriv.groupby(['condition', 'time']).agg(
                                n_subjects=('id', 'nunique'),
                                mean=('dval_dt', 'mean'),
                                sem=('dval_dt', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                            ).reset_index()

                            # Plot derivative condition mean with subject derivatives overlay
                            plt.figure(figsize=(8,5))
                            for cond, cond_df in cond_deriv.groupby('condition'):
                                col = condition_colors.get(str(cond), None)
                                if col is not None:
                                    plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
                                    plt.plot(cond_df['time'], cond_df['mean'], color=col, label=f"{cond} (mean)")
                                else:
                                    plt.fill_between(cond_df['time'], cond_df['mean'] - cond_df['sem'], cond_df['mean'] + cond_df['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
                                    plt.plot(cond_df['time'], cond_df['mean'], label=f"{cond} (mean)")

                            # Overlay faint per-subject derivative lines
                            for (sid, cond), sgroup in subj_deriv.groupby(['id', 'condition']):
                                col = condition_colors.get(str(cond), None)
                                plt.plot(sgroup['time'], sgroup['dval_dt'], alpha=0.2, linewidth=0.6, label=None, color=col)

                            plt.xlabel('Time (s)')
                            plt.ylabel('d(value)/dt')
                            plt.title(f"{exp} : {ev} : {sig} (first derivative)")
                            plt.axvline(0, color='k', linestyle='--')
                            plt.legend()
                            plt.tight_layout()
                            dplotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}_d1.svg")
                            plt.savefig(dplotfile, dpi=300)
                            plt.close()
                        # ---------------------------------------
                        # Diagnostics: per-subject trial counts
                        # ---------------------------------------
                        try:
                            pairs = sig_df[['id', 'condition', 'datetime', 'event','trial']].dropna()
                            pairs['pair'] = pairs['datetime'].astype(str) + '|' + pairs['trial'].astype(str)
                            trial_counts = pairs.groupby(['event','id', 'condition'])['pair'].nunique().reset_index().rename(columns={'pair': 'n_trials'})

                            diag_dir = os.path.join(output_dir, 'diagnostics')
                            os.makedirs(diag_dir, exist_ok=True)
                            diag_file = os.path.join(diag_dir, f"trial_counts_{exp}_{ev}_{sig}.csv")
                            trial_counts.to_csv(diag_file, index=False)

                            
                        except Exception:
                            # Keep pipeline robust; diagnostics are optional
                            pass
                        # ----------------------------
                        # Individual subject trial plots (saved per event)
                        # ----------------------------
                        indiv_dir = os.path.join(ev_out_dir, 'individual_trials', f"{sig}")
                        os.makedirs(indiv_dir, exist_ok=True)

                        for sid in sig_df['id'].dropna().unique():
                            subj_df = sig_df[sig_df['id'] == sid]
                            subj_mean = subject_mean[subject_mean['id'] == sid]
                            # Count unique trials (by datetime x trial) for this subject
                            try:
                                trial_pairs = subj_df[['datetime', 'trial']].dropna().drop_duplicates()
                                n_trials = int(trial_pairs.shape[0])
                            except Exception:
                                n_trials = None
                            plt.figure(figsize=(8,5))
                            for cond in subj_df['condition'].dropna().unique():
                                cond_df = subj_df[subj_df['condition'] == cond]
                                col = condition_colors.get(str(cond), None)
                                # plot subject mean
                                cond_mean = subj_mean[subj_mean['condition'] == cond]
                                if not cond_mean.empty:
                                    plt.plot(cond_mean['time'], cond_mean['val'], color=col, label=f"{cond} (mean)")
                                for block in cond_df['datetime'].dropna().unique():
                                    block_df = cond_df[cond_df['datetime'] == block]
                                    for trial in block_df['trial'].dropna().unique():
                                        trial_df = block_df[block_df['trial'] == trial]
                                        plt.plot(trial_df['time'], trial_df['value'], alpha=0.2, linewidth=0.6, color=col)

                            plt.xlabel('Time (s)')
                            plt.ylabel(args.baseline_correct_method)
                            if n_trials is not None:
                                plt.title(f"{exp} : {ev} : {sig} : Subject {sid} (n trials={n_trials})")
                            else:
                                plt.title(f"{exp} : {ev} : {sig} : Subject {sid}")
                            plt.axvline(0, color='k', linestyle='--')
                            plt.legend()
                            plt.tight_layout()
                            indiv_plotfile = os.path.join(indiv_dir, f"{exp}_{ev}_{sig}_{sid}.svg")
                            plt.savefig(indiv_plotfile, dpi=300)
                            plt.close()
                # ---------------------------------------
                # SCWT congruency contrast: (incongruent - congruent) per condition
                # ---------------------------------------
                # if str(exp).upper() == 'SCWT':
                #     cong_dir = os.path.join(exp_out_dir, 'congruency_contrast')
                #     os.makedirs(cong_dir, exist_ok=True)
                #     for sig in exp_df['signal_type'].dropna().unique():
                #         sig_all = exp_df[exp_df['signal_type'] == sig]
                #         # Identify congruent/incongruent correct cue events
                #         cong_mask = sig_all['event'].astype(str).str.contains('cue.*congruent.*correct', case=False, na=False)
                #         incong_mask = sig_all['event'].astype(str).str.contains('cue.*incongruent.*correct', case=False, na=False)
                #         cong_df = sig_all[cong_mask]
                #         incong_df = sig_all[incong_mask]
                #         if cong_df.empty or incong_df.empty:
                #             continue

                #         # Subject-level mean traces for congruent and incongruent
                #         subj_cong = cong_df.groupby(['id', 'condition', 'time'])['value'].mean().reset_index().rename(columns={'value': 'congruent'})
                #         subj_incong = incong_df.groupby(['id', 'condition', 'time'])['value'].mean().reset_index().rename(columns={'value': 'incongruent'})

                #         # Inner join by id/condition/time to ensure aligned samples
                #         subj_merge = pd.merge(subj_incong, subj_cong, on=['id', 'condition', 'time'], how='inner')
                #         if subj_merge.empty:
                #             continue
                #         subj_merge['val'] = subj_merge['incongruent'] - subj_merge['congruent']

                #         # Compute metrics from contrast timeseries per subject/condition
                #         metrics_rows = []
                #         for (sid, cond), sgroup in subj_merge.groupby(['id', 'condition']):
                #             m = compute_timeseries_metrics(sgroup[['time', 'val']])
                #             for k, v in m.items():
                #                 metrics_rows.append({
                #                     'expt': exp,
                #                     'event': 'congruency_contrast',
                #                     'signal': sig,
                #                     'id': sid,
                #                     'condition': cond,
                #                     'metric': k,
                #                     'value': v,
                #                 })

                #         if metrics_rows:
                #             metrics_df = pd.DataFrame(metrics_rows)
                #             metrics_dir = os.path.join(output_dir, 'metrics_by_timeseries')
                #             os.makedirs(metrics_dir, exist_ok=True)
                #             metrics_csv = os.path.join(metrics_dir, 'timeseries_metrics.csv')
                #             metrics_df.to_csv(metrics_csv, index=False, mode='a', header=not os.path.exists(metrics_csv))

                #         # Group condition mean and SEM across subjects for contrast
                #         cond_mean = subj_merge.groupby(['condition', 'time']).agg(
                #             n_subjects=('id', 'nunique'),
                #             mean=('val', 'mean'),
                #             sem=('val', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                #         ).reset_index()

                #         # Plot contrast per condition with subject overlays
                #         plt.figure(figsize=(6, 4.5))
                #         plt.axvline(0, color='k', linestyle='--')
                #         for cond, cdf in cond_mean.groupby('condition'):
                #             col = condition_colors.get(str(cond), None)
                #             if col is not None:
                #                 plt.fill_between(cdf['time'], cdf['mean'] - cdf['sem'], cdf['mean'] + cdf['sem'], linewidth=0, color=col, alpha=0.3, label=f"{cond} (sem)")
                #                 plt.plot(cdf['time'], cdf['mean'], color=col, label=f"{cond} (mean)")
                #             else:
                #                 plt.fill_between(cdf['time'], cdf['mean'] - cdf['sem'], cdf['mean'] + cdf['sem'], linewidth=0, alpha=0.3, label=f"{cond} (sem)")
                #                 plt.plot(cdf['time'], cdf['mean'], label=f"{cond} (mean)")

                #         # overlay subject contrast traces
                #         for (sid, cond), sgroup in subj_merge.groupby(['id', 'condition']):
                #             col = condition_colors.get(str(cond), None)
                #             plt.plot(sgroup['time'], sgroup['val'], alpha=0.25, linewidth=0.6, label=None, color=col)

                #         plt.xlabel('Time (s)')
                #         plt.ylabel(args.baseline_correct_method)
                #         plt.title(f"{exp} : congruency_contrast : {sig}")
                #         plt.legend()
                #         plt.tight_layout()
                #         out_file = os.path.join(cong_dir, f"{exp}_congruency_contrast_{sig}.svg")
                #         plt.savefig(out_file, dpi=300)
                #         plt.close()
    except Exception:
        # don't crash after main; print traceback for debugging
        print("Error computing experiment-level overlays:")
        traceback.print_exc()

    print(f"Data successfully exported to {output_dir}")


if __name__ == "__main__":
    main()
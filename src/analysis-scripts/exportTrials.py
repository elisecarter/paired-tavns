import os
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import datetime
import argparse
import sys
import traceback

def epoch_by_event(block_data, event, event_df, pre_event_dur=5, post_event_dur=5, block_cfg=None, baseline_method='zscore'):
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

        for sig in block_data.columns.difference(['Timestamps', 'Event', 'nSeq']):
            values = block_data[sig].iloc[start_ind:end_ind].to_numpy(dtype=float)

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
            elif baseline_method == 'subtract_mean':
                z_values = values - BL_mean
            else:
                # unknown method: default to zscore
                z_values = (values - BL_mean) / BL_std

            # Extract features on post-event window (extract_features expects numpy arrays)
            features = extract_features(z_values, rel_time, sig)

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
                'value': np.round(z_values, 3)
            })

            # explode arrays into long-form rows
            # Use explode for both columns and ensure numeric scalars (not arrays/objects) for plotting
            ts_long = ts_df.explode(['time', 'value']).reset_index(drop=True)
            # coerce types to numeric to avoid object/array cells that matplotlib may interpret oddly
            ts_long['time'] = pd.to_numeric(ts_long['time'], errors='coerce')
            ts_long['value'] = pd.to_numeric(ts_long['value'], errors='coerce')
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
                'baseline_valid': baseline_flag
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
    peak_time = time[trial_data.argmax()]
    trough_time = time[trial_data.argmin()]
    dt = float(np.mean(np.diff(time))) if len(time) > 1 else 0.0
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
        'time_to_peak': peak_time,
        'time_to_trough': trough_time,
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
    parser.add_argument('--baseline-method', choices=['zscore', 'subtract_mean'], default='zscore', help='Baseline correction method to apply to epochs')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = os.path.join(args.output_dir, today)
    start_date = args.start_date
    end_date = args.end_date
    dry_run = args.dry_run
    os.makedirs(output_dir, exist_ok=True)

    # master accumulator for all block timeseries to compute experiment-level averages
    master_timeseries = []
    # condition colors (hex) — used for plotting
    condition_colors = {
        'sham': '#68689a',
        'taVNS': '#6666ff'
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
                    if 'PLRT' in experiment_type or 'SCWT' in experiment_type:
                        pre, post = 1, 4
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
                            if 'stim' in event.lower():
                                if "_" in event: # strip condition from stimulus event
                                    event_prefix = event.split('_')[0]
                                    df_events.at[idx, 'event'] = event_prefix


                        response_mask = df_events['event'].str.contains('response_', case=False, na=False)
                        df_events = df_events.loc[~response_mask]
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
                        trial_data = epoch_by_event(ts_data, event, event_df, pre_event_dur=pre, post_event_dur=post, block_cfg=block_cfg, baseline_method=args.baseline_method)
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

                        # Save raw feature rows
                        csv_file = os.path.join(output_dir, "features-table.csv")
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

                        # condition mean and sem across subjects
                        condition_mean = subject_mean.groupby(['experiment', 'signal_type', 'event', 'condition', 'time']).agg(
                            n_subjects=('id', 'nunique'),
                            mean=('val', 'mean'),
                            sem=('val', lambda x: x.std(ddof=0) / np.sqrt(x.nunique() if x.nunique() > 0 else 1))
                        ).reset_index()

                        
                    
                        # plotting condition mean with subject traces overlay
                        plt.figure(figsize=(8,5))
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

                        # overlay subject traces
                        if not subject_mean.empty:
                            for (sid, cond), sgroup in subject_mean.groupby(['id', 'condition']):
                                col = condition_colors.get(str(cond), None)
                                plt.plot(sgroup['time'], sgroup['val'], alpha=0.2, linewidth=0.6, label=None, color=col)

                        plt.xlabel('Time (s)')
                        plt.ylabel(args.baseline_method)
                        plt.title(f"{exp} : {ev} : {sig}")
                        plt.axvline(0, color='k', linestyle='--')
                        plt.legend()
                        plt.tight_layout()
                        plotfile = os.path.join(ev_out_dir, f"{exp}_{ev}_{sig}.png")
                        plt.savefig(plotfile, dpi=300)
                        plt.close()
                        # ----------------------------
                        # Individual subject trial plots (saved per event)
                        # ----------------------------
                        indiv_dir = os.path.join(ev_out_dir, 'individual_trials', f"{sig}")
                        os.makedirs(indiv_dir, exist_ok=True)

                        for sid in sig_df['id'].dropna().unique():
                            subj_df = sig_df[sig_df['id'] == sid]
                            subj_mean = subject_mean[subject_mean['id'] == sid]
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
                            plt.ylabel(args.baseline_method)
                            plt.title(f"{exp} : {ev} : {sig} : Subject {sid}")
                            plt.axvline(0, color='k', linestyle='--')
                            plt.legend()
                            plt.tight_layout()
                            indiv_plotfile = os.path.join(indiv_dir, f"{exp}_{ev}_{sig}_{sid}.png")
                            plt.savefig(indiv_plotfile, dpi=300)
                            plt.close()
    except Exception:
        # don't crash after main; print traceback for debugging
        print("Error computing experiment-level overlays:")
        traceback.print_exc()

    print(f"Data successfully exported to {output_dir}")


if __name__ == "__main__":
    main()
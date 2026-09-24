"""
GUI + batch runner for manually annotating the first fixation-cross frame in Pupil Labs Neon scene-camera (world view) recordings.

Single recording/block:
    python annotateFixationCross.py --recording-dir "/path/to/neon/recording"
    python annotateFixationCross.py --block-path "/path/to/session/block"

Batch over a data directory (skips blocks already annotated unless --force):
    python annotateFixationCross.py --data-dir /path/to/Data --start-date 20250701
    python annotateFixationCross.py --subject ERC06 --force

--block-path / --data-dir auto-locate the closest same-day Neon recording
directory under --backup-root and save the output JSON alongside the block
instead of on the backup drive. `preprocess-timeseries.py` only ever reads
the resulting annotation JSON and never launches this GUI itself.

GUI Controls:
    Left / Right        : step 1 frame backward / forward
    Shift+Left/Right     : step 10 frames backward / forward
    Up / Down           : step 100 frames backward / forward
    Home / End          : jump to first / last frame
    Slider              : drag to scrub to any frame
    m                   : mark current frame as first fixation cross
    s                   : save marked frame + timestamp and close
    q                   : quit without saving
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Free up keys that matplotlib binds by default (save/quit/back/forward/home)
# so they don't fight with our navigation shortcuts below.
plt.rcParams['keymap.back'] = []
plt.rcParams['keymap.forward'] = []
plt.rcParams['keymap.home'] = []
plt.rcParams['keymap.save'] = []
plt.rcParams['keymap.quit'] = []

SCENE_VIDEO_NAME = "Neon Scene Camera v1 ps1.mp4"
SCENE_TIME_NAME = "Neon Scene Camera v1 ps1.time"
PUPIL_LABS_BACKUP_ROOT = Path(r"/Volumes/WHSynology/BIOElectricsLab/Elise/pupil labs backup")


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


def find_closest_neon_recording_path(reference_path, backup_root=PUPIL_LABS_BACKUP_ROOT):
    """Find nearest same-day Neon backup directory to a block path (returns str path or None)."""
    reference_path = Path(reference_path)
    reference_dt = (
        _parse_datetime_from_text(reference_path.name)
        or _parse_datetime_from_text(reference_path.parent.name)
        or _parse_datetime_from_text(str(reference_path))
    )
    if reference_dt is None or not Path(backup_root).exists():
        return None

    candidates = []
    for current_root, dir_names, _ in os.walk(backup_root):
        for dir_name in dir_names:
            dt = _parse_datetime_from_text(dir_name)
            if dt is not None:
                candidates.append((dt, Path(current_root) / dir_name))

    same_day = [(dt, p) for dt, p in candidates if dt.date() == reference_dt.date()]
    if not same_day:
        return None

    closest_dt, closest_path = min(same_day, key=lambda item: abs((item[0] - reference_dt).total_seconds()))
    if abs((closest_dt - reference_dt).total_seconds()) > 60:
        return None
    return str(closest_path)


def load_scene_video(recording_dir):
    """Open the Neon scene-camera video and its per-frame timestamps."""
    recording_dir = Path(recording_dir)
    video_path = recording_dir / SCENE_VIDEO_NAME
    time_path = recording_dir / SCENE_TIME_NAME
    if not video_path.exists():
        raise FileNotFoundError(f"Scene camera video not found: {video_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Scene camera timestamps not found: {time_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    timestamps_ns = np.fromfile(time_path, dtype=np.int64)
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames_video, timestamps_ns.size) if timestamps_ns.size else n_frames_video
    if n_frames <= 0:
        raise RuntimeError(f"No frames available in scene camera video: {video_path}")

    return cap, timestamps_ns, n_frames, video_path


def scene_camera_available(recording_dir):
    """Return whether the recording contains the scene video and frame timestamps."""
    recording_dir = Path(recording_dir)
    return (
        (recording_dir / SCENE_VIDEO_NAME).is_file()
        and (recording_dir / SCENE_TIME_NAME).is_file()
    )


class FixationTaggerGUI:
    """Matplotlib-based frame scrubber for tagging the first fixation-cross frame."""

    def __init__(self, recording_dir, output_dir=None, output_prefix=None):
        self.recording_dir = Path(recording_dir)
        self.cap, self.timestamps_ns, self.n_frames, self.video_path = load_scene_video(self.recording_dir)
        self.idx = 0
        self.marked_idx = None
        self._last_read_idx = -2  # forces an initial seek on first frame read

        output_dir = Path(output_dir) if output_dir else self.recording_dir
        filename = f"{output_prefix}_fixationCrossTimestamp.json" if output_prefix else "fixationCrossTimestamp.json"
        self.output_path = output_dir / filename

        self.fig, self.ax_img = plt.subplots(figsize=(11, 8.5))
        plt.subplots_adjust(bottom=0.22, top=0.90)
        self.im = self.ax_img.imshow(self._read_frame(0))
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.fig.suptitle(self._title_text())

        help_text = (
            "Left/Right: +/-1 frame   Shift+Left/Right: +/-10   Up/Down: +/-100   Home/End: first/last\n"
            "m: mark current frame   s: save & close   q: quit without saving"
        )
        self.fig.text(0.01, 0.955, help_text, fontsize=8, va='top')

        ax_slider = plt.axes((0.15, 0.12, 0.7, 0.03))
        self.slider = Slider(ax_slider, 'Frame', 0, max(self.n_frames - 1, 0), valinit=0, valstep=1)
        self.slider.on_changed(lambda val: self._set_idx(int(val)))

        mark_ax = plt.axes((0.15, 0.03, 0.2, 0.05))
        save_ax = plt.axes((0.40, 0.03, 0.2, 0.05))
        quit_ax = plt.axes((0.65, 0.03, 0.2, 0.05))
        Button(mark_ax, 'Mark (m)').on_clicked(lambda e: self.mark_current())
        Button(save_ax, 'Save & Close (s)').on_clicked(lambda e: self.save_and_close())
        Button(quit_ax, 'Quit (q)').on_clicked(lambda e: plt.close(self.fig))
        # Keep references so buttons stay responsive (matplotlib needs a live reference).
        self._buttons = [mark_ax, save_ax, quit_ax]

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _read_frame(self, idx):
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        if idx != self._last_read_idx + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        self._last_read_idx = idx
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _timestamp_for(self, idx):
        if self.timestamps_ns.size > idx:
            return int(self.timestamps_ns[idx])
        return None

    def _title_text(self):
        ts_ns = self._timestamp_for(self.idx)
        ts_str = f"{ts_ns / 1e9:.3f}s (unix)" if ts_ns is not None else "n/a"
        marked_str = f" | marked frame: {self.marked_idx}" if self.marked_idx is not None else ""
        return f"Frame {self.idx}/{self.n_frames - 1}  |  ts: {ts_str}{marked_str}"

    def _refresh(self):
        self.im.set_data(self._read_frame(self.idx))
        self.fig.suptitle(self._title_text())
        if int(self.slider.val) != self.idx:
            self.slider.eventson = False
            self.slider.set_val(self.idx)
            self.slider.eventson = True
        self.fig.canvas.draw_idle()

    def _set_idx(self, new_idx):
        self.idx = int(np.clip(new_idx, 0, self.n_frames - 1))
        self._refresh()

    def mark_current(self):
        self.marked_idx = self.idx
        print(f"Marked frame {self.idx} as first fixation cross (ts_ns={self._timestamp_for(self.idx)})")
        self._refresh()

    def save_and_close(self):
        if self.marked_idx is None:
            print("No frame marked yet - press 'm' to mark a frame before saving.")
            return
        ts_ns = self._timestamp_for(self.marked_idx)
        result = {
            "recording_dir": str(self.recording_dir),
            "video_file": str(self.video_path),
            "frame_index": int(self.marked_idx),
            "timestamp_ns": ts_ns,
            "timestamp_unix_sec": ts_ns / 1e9 if ts_ns is not None else None,
            "tagged_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved fixation cross tag to {self.output_path}")
        plt.close(self.fig)

    def _on_key(self, event):
        key = event.key or ''
        steps = {
            'left': -1, 'right': 1,
            'shift+left': -10, 'shift+right': 10,
            'up': 100, 'down': -100,
        }
        if key in steps:
            self._set_idx(self.idx + steps[key])
        elif key == 'home':
            self._set_idx(0)
        elif key == 'end':
            self._set_idx(self.n_frames - 1)
        elif key == 'm':
            self.mark_current()
        elif key == 's':
            self.save_and_close()
        elif key == 'q':
            plt.close(self.fig)

    def run(self):
        plt.show()
        self.cap.release()


def tag_already_exists(block_path, block_str):
    """Return True if a valid fixation-cross tag JSON already exists for this block."""
    tag_path = os.path.join(block_path, f"{block_str}_fixationCrossTimestamp.json")
    if not os.path.exists(tag_path):
        return False
    with open(tag_path, 'r') as f:
        existing_tag = json.load(f)
    return existing_tag.get('timestamp_unix_sec') is not None


def run_batch(args):
    """Walk subject/session/block directories and tag any untagged pupil-recording block."""
    backup_root = Path(args.backup_root)

    for subject in os.listdir(args.data_dir):
        subject_path = os.path.join(args.data_dir, subject)
        if not os.path.isdir(subject_path) or subject.startswith("test"):
            continue
        if args.subject and subject != args.subject:
            continue

        for session in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session)
            try:
                sess_int = int(session)
            except ValueError:
                continue
            if not os.path.isdir(session_path) or not (args.start_date <= sess_int <= args.end_date):
                continue

            for block in os.listdir(session_path):
                block_path = os.path.join(session_path, block)
                if not os.path.isdir(block_path):
                    continue

                cfg_path = os.path.join(block_path, f"{block}_config.json")
                block_cfg = json.load(open(cfg_path, 'r')) if os.path.exists(cfg_path) else {}
                if not block_cfg.get('record_pupil', False):
                    continue

                if tag_already_exists(block_path, block) and not args.force:
                    print(f"Skipping {block_path} — already tagged (use --force to retag)")
                    continue

                neon_recording_path = find_closest_neon_recording_path(block_path, backup_root=backup_root)
                if neon_recording_path is None:
                    print(f"No matching Neon recording found for {block_path}; skipping")
                    continue
                if not scene_camera_available(neon_recording_path):
                    print(
                        f"No scene camera video/timestamps found for {block_path} "
                        f"under {neon_recording_path}; skipping"
                    )
                    continue

                print(f"Tagging {block_path} using Neon recording {neon_recording_path}")
                FixationTaggerGUI(neon_recording_path, output_dir=block_path, output_prefix=block).run()


def main():
    parser = argparse.ArgumentParser(description="Tag the first fixation-cross frame in Pupil Labs Neon recordings")
    parser.add_argument('--recording-dir', help="Path to a single Neon recording directory (contains 'Neon Scene Camera v1 ps1.mp4')")
    parser.add_argument('--block-path', help="Path to a single task block directory; auto-locates the closest same-day Neon recording under --backup-root")
    parser.add_argument('--data-dir', default=r"/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data", help='Batch mode: top-level data directory')
    parser.add_argument('--start-date', type=int, default=20250701, help='Batch mode: start session (YYYYMMDD)')
    parser.add_argument('--end-date', type=int, default=20251031, help='Batch mode: end session (YYYYMMDD)')
    parser.add_argument('--subject', help='Batch mode: only process this subject folder')
    parser.add_argument('--force', action='store_true', help='Retag blocks even if already tagged')
    parser.add_argument('--backup-root', default=str(PUPIL_LABS_BACKUP_ROOT), help="Root directory to search for Neon recordings")
    args = parser.parse_args()

    if args.recording_dir or args.block_path:
        output_dir = None
        output_prefix = None
        if args.recording_dir:
            recording_dir = args.recording_dir
        else:
            recording_dir = find_closest_neon_recording_path(args.block_path, backup_root=Path(args.backup_root))
            if recording_dir is None:
                raise SystemExit("Could not auto-locate a Neon recording for the given block path. Pass --recording-dir directly.")
            if not scene_camera_available(recording_dir):
                raise SystemExit(
                    f"No scene camera video/timestamps found under {recording_dir}; skipping {args.block_path}."
                )
            output_dir = args.block_path
            output_prefix = Path(args.block_path).name
            print(f"Matched Neon recording directory: {recording_dir}")

        if not scene_camera_available(recording_dir):
            raise SystemExit(
                f"No scene camera video/timestamps found under {recording_dir}; skipping."
            )
        FixationTaggerGUI(recording_dir, output_dir=output_dir, output_prefix=output_prefix).run()
    else:
        run_batch(args)


if __name__ == "__main__":
    main()

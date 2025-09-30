import os
import sys
import json
import random
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "Data")
CALIBRATE_SCRIPT = r"calibrateStimulation.py"
EXPERIMENT_SCRIPTS = {
    "Stroop Color Word Task": r"stroopTask.py",
    "Stroop Squared": r"stroopSquared.py",
    "PLRT": r"PLRT.py"
}

class SessionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Start New Session")
        self.order = ""
        self.session_dir = ""
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.calibration_results = {}  # Will become dict mapping frequency to calibration data
        self.condition = ""
        self.default_percentage = '150%'  # Default to 150% of threshold
        self.config_path = None

        # New variables for calibration frequency options
        self.calib_freq_30_var = tk.BooleanVar(value=True)
        self.calib_freq_95_var = tk.BooleanVar(value=False)
        # Variable for selecting which calibration frequency to use later
        self.stim_freq_var = tk.StringVar()

        self.create_session_window()

    def create_session_window(self):
        self.title("Start New Session")
        self.geometry("400x500")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.quit)

        frame = ttk.Frame(self, padding=50)
        frame.pack(fill="both", expand=True)
        row = 0
        # Participant ID input
        ttk.Label(frame, text="Participant ID:").grid(row=row, column=0, sticky="w", pady=(0, 10))
        self.participant_id_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.participant_id_var).grid(row=row, column=1, sticky="ew", pady=(0, 10))
        row += 1
        
        # Condition Order dropdown
        ttk.Label(frame, text="Condition Order:").grid(row=row, column=0, sticky="w", pady=(0, 10))
        self.order_var = tk.StringVar()
        order_options = ["STTS", "TSST"]
        order_cb = ttk.Combobox(frame, textvariable=self.order_var, state="readonly", values=order_options)
        order_cb.grid(row=row, column=1, sticky="ew", pady=(0, 10))
        order_cb.current(0)
        row += 1

        # New: Calibration Frequency options as checkboxes
        ttk.Label(frame, text="Stimulation Frequency (Hz):").grid(row=row, column=0, sticky="w", pady=(0, 10))
        freq_frame = ttk.Frame(frame)
        freq_frame.grid(row=row, column=1, sticky="w", pady=(0, 10))
        ttk.Checkbutton(freq_frame, text="30", variable=self.calib_freq_30_var).pack(side="left")
        ttk.Checkbutton(freq_frame, text="100", variable=self.calib_freq_95_var).pack(side="left")
        row += 1
        
        # Toggle for Triggering Stimulation
        self.trigger_stim_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Trigger Stimulation", variable=self.trigger_stim_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        # # Toggle for Stream Events to LSL
        # self.stream_events_var = tk.BooleanVar(value=False)
        # ttk.Checkbutton(frame, text="Stream Events to LSL", variable=self.stream_events_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        # row += 1

        # Acquisition Mode dropdown (LSL or direct)
        ttk.Label(frame, text="Acquisition Mode:").grid(row=row, column=0, sticky="w", pady=(0, 10))
        self.acquisition_mode_var = tk.StringVar(value="LSL")
        acq_options = ["LSL", "direct"]
        ttk.Combobox(frame, textvariable=self.acquisition_mode_var, state="readonly", values=acq_options).grid(row=row, column=1, sticky="ew", pady=(0, 10))
        row += 1

        # Toggle for Record Pupil Diameter
        self.record_pupil_var = tk.BooleanVar(value=False)
        pupil_cb = ttk.Checkbutton(frame, text="Record Pupil Diameter", variable=self.record_pupil_var, command=self.update_neon_IP_state)
        pupil_cb.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        # Neon IP Address
        self.neon_ip_var = tk.StringVar(value="10.168.182.81")  # Default IP for Neon device
        self.neon_ip_entry = ttk.Entry(frame, textvariable=self.neon_ip_var)
        self.neon_ip_entry.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.neon_ip_entry.state(["disabled"])
        row += 1

        # Toggle for Record from BITalino
        self.record_bitalino_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Record from BITalino", variable=self.record_bitalino_var).grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        # BITalino MAC Address
        ttk.Label(frame, text="BITalino MAC:").grid(row=row, column=0, sticky="w", pady=(0,10))
        self.bitalino_mac_var = tk.StringVar(value="20:19:07:00:80:63")
        ttk.Entry(frame, textvariable=self.bitalino_mac_var).grid(row=row, column=1, sticky="ew", pady=(0, 10))
        row += 1

        # BITalino Sampling Rate and Channels
        ttk.Label(frame, text="BITalino srate/ch:").grid(row=row, column=0, sticky="w", pady=(0,10))
        self.bitalino_srate_var = tk.StringVar(value="1000")
        self.bitalino_ch_var = tk.StringVar(value="1,2,3")
        ttk.Entry(frame, textvariable=self.bitalino_srate_var).grid(row=row, column=1, sticky="ew", pady=(0, 10))
        ttk.Entry(frame, textvariable=self.bitalino_ch_var).grid(row=row, column=1, sticky="e", pady=(0, 10))
        row += 1
        
        # Submit button to start the session
        submit_btn = ttk.Button(frame, text="Start Session", command=self.submit_session)
        submit_btn.grid(row=row, column=0, columnspan=2, pady=(20, 0))

        frame.columnconfigure(1, weight=1)

    def update_neon_IP_state(self):
        if self.record_pupil_var.get():
            self.neon_ip_entry.state(["!disabled"])
        else:
            self.neon_ip_entry.state(["disabled"])
    
    def run_calibration(self, calib_freqs):
        import calibrateStimulation as calib
        # calibration order determined from order string's last two characters
        try:
            if self.order[-2:].lower() == "st":
                calib_order = ["sham", "taVNS"]
            elif self.order[-2:].lower() == "ts":
                calib_order = ["taVNS", "sham"]
            else:
                calib_order = ["sham", "taVNS"]
        except IndexError:
            calib_order = ["sham", "taVNS"]

        for freq in calib_freqs:
            print(f"Running calibration for frequency: {freq} Hz")
            calibrator = calib.Calibrator(
                outdir=self.session_dir,
                conditions=calib_order,
                frequency=freq
            )
            results = calibrator.run()
            self.calibration_results[str(freq)] = results

        # update config with results and calibration frequencies used
        if self.config_path is None:
            raise ValueError("Configuration path is not set.")
        with open(self.config_path, "r+") as f:
            cfg = json.load(f)
            cfg.update({
                "calibration results": self.calibration_results
            })
            f.seek(0)
            f.truncate()
            json.dump(cfg, f, indent=2)
        self._build_selections_frame(cfg)

    def submit_session(self):
        pid = self.participant_id_var.get().strip()
        order = self.order_var.get().strip()
        if not pid:
            messagebox.showerror("Input Error", "Please enter a Participant ID.")
            return
        if not order:
            messagebox.showerror("Input Error", "Please select a Condition Order.")
            return

        self.order = order
        self.session_dir = os.path.join(BASE_DIR, pid, self.date_str)
        os.makedirs(self.session_dir, exist_ok=True)
        self.config_path = os.path.join(self.session_dir, "session_config.json")
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
            except json.JSONDecodeError:
                cfg = {}
        else:
            cfg = {}

        # Save selected calibration frequencies as a list
        calib_freqs = []
        if self.calib_freq_30_var.get():
            calib_freqs.append(30)
        if self.calib_freq_95_var.get():
            calib_freqs.append(100)
        if not calib_freqs:
            messagebox.showerror("Frequency Error", "Select at least one calibration frequency.")
            return

        cfg.update({
            "ID": pid,
            "session_dir": self.session_dir,
            "date": self.date_str,
            "order": order,
            "calibration_frequency": calib_freqs,
            "trigger_stim": self.trigger_stim_var.get(),
            "record_pupil": self.record_pupil_var.get(),
            "record_bitalino": self.record_bitalino_var.get(),
            "acquisition_mode": self.acquisition_mode_var.get(),
            "neon_ip": self.neon_ip_var.get(),
            "bitalino_mac": self.bitalino_mac_var.get(),
            "bitalino_srate": int(self.bitalino_srate_var.get()),
            "bitalino_channels": [int(x) for x in self.bitalino_ch_var.get().split(",") if x.strip()]
        })
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Check existing calibration results for selected frequencies and run calibrations for any missing ones.
        if "calibration results" not in cfg:
            if cfg['trigger_stim']:
                messagebox.showinfo("Calibration", "No calibration results found. Running calibration...")
                self.run_calibration(calib_freqs)
            else:
                self._build_selections_frame(cfg) # need to call if not running calibration
        else:
            self.calibration_results = cfg.get("calibration results", {})
            missing_freqs = [freq for freq in calib_freqs if str(freq) not in self.calibration_results]
            if missing_freqs:
                # If there are missing frequencies, run calibration for those
                self.run_calibration(missing_freqs)

            self._build_selections_frame(cfg) # need to call if not running calibration


    def prompt_additional_calibration(self):
        new_freq = simpledialog.askstring("Calibration Frequency", "Enter new calibration frequency (Hz):")

        # Overwrite the calibration frequency selection and run calibration
        # Here we simply run calibration with the new value.
        self.run_calibration([new_freq])

    def _build_selections_frame(self, cfg):
        for w in self.winfo_children():
            w.destroy()
        self.block_idx = 0  # Reset block index for new session
        frm = ttk.Frame(self, padding=50)
        frm.pack(fill="both", expand=True)
        
        # Add button for recalibration
        ttk.Button(frm, text="Recalibrate", command=self.prompt_additional_calibration).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        next_row = 1
        
        # Select Stimulation Frequency
        ttk.Label(frm, text="Stimulation Frequency (Hz):").grid(row=1, column=0, sticky="w", pady=(0, 10))
        freq_options = list(self.calibration_results.keys())
        if cfg['trigger_stim']:
            self.stim_freq_var.set(freq_options[0])
        else:
            self.stim_freq_var.set('')
        calib_cb = ttk.Combobox(frm, textvariable=self.stim_freq_var, state="readonly", values=freq_options)
        calib_cb.grid(row=1, column=1,  columnspan=2,sticky="e", pady=(0, 10))
        if not cfg['trigger_stim']:
            calib_cb.config(state="disabled")
        next_row +=1

        # Select Experiment
        ttk.Label(frm, text="Select Experiment:").grid(row=next_row, column=0, sticky="w",pady=(0, 10))
        self.exp_var = tk.StringVar()
        cb = ttk.Combobox(
            frm,
            textvariable=self.exp_var,
            state="readonly",
            values=list(EXPERIMENT_SCRIPTS.keys())
        )
        cb.grid(row=next_row, column=1, columnspan=2, pady=(0, 10), sticky="e")
        frm.columnconfigure(1, weight=1)

        # Number of Blocks
        ttk.Label(frm, text="Number of Blocks:").grid(row=next_row + 1, column=0, pady=(0, 10), sticky="w")
        self.blocks_var = tk.StringVar(value="4")  # Default to 4 blocks
        ttk.Entry(frm, textvariable=self.blocks_var).grid(row=next_row + 1, column=1, columnspan=2, sticky="e",pady=(0, 10))

        # Default Percentage of PT Dropdown
        ttk.Label(frm, text="% of PT:").grid(row=next_row + 2, column=0, sticky="w",pady=(0, 10))
        self.percentage_var = tk.StringVar(value=self.default_percentage)
        percentage_options = ["100%", "150%", "200%"]
        cb = ttk.Combobox(frm, textvariable=self.percentage_var, state="readonly", values=percentage_options)
        cb.grid(row=next_row + 2, column=1, sticky="e",columnspan=2,pady=(0, 10))
        if not cfg['trigger_stim']:
            cb.config(state="disabled")

        # Next Button
        ttk.Button(frm, text="Next", command=lambda: self._validate_selections(cfg)).grid(row=next_row + 3, column=0, columnspan=2, pady=(0, 10))

    def _validate_selections(self,cfg):
        exp = self.exp_var.get().strip()
        if not exp:
            messagebox.showerror("Invalid Input", "Please select an experiment.")
            return

        blocks_value = self.blocks_var.get().strip()
        try:
            self.num_blocks = int(blocks_value)
            if self.num_blocks <= 0:
                raise ValueError("Number of blocks must be positive.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter a valid number of blocks: {e}")
            return

        stim_freq_value = self.stim_freq_var.get().strip()
        
        if cfg['trigger_stim']:
            try:
                self.stim_freq = float(stim_freq_value)
                if self.stim_freq < 0:
                    raise ValueError("Stimulation frequency cannot be negative.")
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter a valid stimulation frequency: {e}")
                return
            

            percentage_value = self.percentage_var.get().strip()
            if not percentage_value.endswith("%"):
                messagebox.showerror("Invalid Input", "% of PT must end with '%' (e.g. 150%).")
                return
            try:
                num = float(percentage_value[:-1])
                if num <= 0:
                    raise ValueError("Percentage must be greater than 0.")
                self.default_percentage = percentage_value
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter a valid percentage: {e}")
                return
        else:
            self.stim_freq = 0
            self.default_percentage= "0%"

        self._build_experiment_frame()

    def _build_experiment_frame(self):
        for w in self.winfo_children():
            w.destroy()
        frm = ttk.Frame(self, padding=25)
        frm.pack(fill="x", expand=True)

        # Ask if there should be a practice block
        practice = messagebox.askyesno("Practice Block", "Would you like to include a practice block?")
        if practice:
            self.block_idx = -1

        self.info_lbl = ttk.Label(frm, text="")
        self.info_lbl.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        ttk.Button(frm, text="Start Block", command=self.start_block).grid(row=3, column=0, columnspan=2, pady=5)
        self._update_info()

    def _update_info(self):
        # Use the calibration data from the selected frequency
        with open(self.config_path, "r") as f:   # type: ignore
            cfg = json.load(f)
        calib_data = self.calibration_results.get(self.stim_freq_var.get(), {})
        if self.block_idx >= self.num_blocks:
            # close the frame
            for w in self.winfo_children():
                w.destroy()
            if messagebox.askyesno("Complete", "All blocks complete. Would you like to run another experiment?"):
                self._build_selections_frame(cfg)
            else:
                # Cleanup the config file and exit by removing specific json fields.
                try:
                    with open(self.config_path, "r+") as f: # type: ignore
                        cfg = json.load(f)
                    keys_to_remove = [
                        "condition",
                        "current_mA",
                        "percentage of PT",
                        "block_no",
                        "stim_freq",
                        "percent_PT",
                        "calibration_frequency"
                    ]
                    for key in keys_to_remove:
                        cfg.pop(key, None)
                    with open(self.config_path, "w") as f: # type: ignore
                        json.dump(cfg, f, indent=2)
                except Exception as e:
                    messagebox.showerror("Config Error", f"Failed to update config: {e}")
                self.quit()
            return
        
        
        cond_char = self.order[self.block_idx]
        if cond_char.upper() == "S":
            condition = "sham"
        elif cond_char.upper() == "T":
            condition = "taVNS"
        else:
            condition = "unknown"

        if self.block_idx < 0:
            txt = "Practice Block - TURN OFF STIMULATOR."
            self.condition = "practice"
            self.current_amplitude = 0.0
            self.percentage  = "0%"
        elif not cfg['trigger_stim']:
            txt = (f"Block {self.block_idx + 1}/{self.num_blocks} - "
                   f"Condition: {condition}")
            self.current_amplitude = 0.0
            self.stim_freq = 0
            self.percentage = "0%"
        else:
            if condition not in calib_data:
                messagebox.showerror("Error", f"No calibration results found for condition '{condition}'.")
                return

            self.condition = condition
            ratings = calib_data.get(condition, {}).get("perceived rating", {})
            current_values = calib_data.get(condition, {}).get("calculated_currents", {})
            options = ["200%", "175%", "150%", "125%", "100%"]
            try:
                start_idx = options.index(self.default_percentage)
            except ValueError:
                start_idx = options.index("150%")
            search_order = options[start_idx:]
            for percentage in search_order:
                if percentage not in ratings or percentage not in current_values:
                    continue
                current = current_values[percentage]
                if current > 5.0 or ratings.get(percentage, 0) > 7:
                    continue
                break
            else:
                percentage = search_order[-1]
                current = current_values.get(percentage, 0.0)

            self.percentage = percentage
            self.current_amplitude = current
            txt = (f"Block {self.block_idx + 1}/{self.num_blocks} - "
                   f"Condition: {condition} - {percentage} Amp: {self.current_amplitude:.2f} mA - {self.stim_freq} Hz")

        self.info_lbl.config(text=txt)

        try:
            with open(self.config_path, "r+") as f:   # type: ignore
                cfg = json.load(f)
                cfg.update({
                    "condition": self.condition,
                    "current_mA": self.current_amplitude,
                    "stim_freq": self.stim_freq,
                    "percent_PT": self.percentage,
                    "block_no": self.block_idx + 1
                })
                f.seek(0)
                f.truncate()
                json.dump(cfg, f, indent=2)
        except Exception as e:
            messagebox.showerror("Config Error", f"Failed to update config: {e}")
        return

    def start_block(self):
        exp = self.exp_var.get()
        script = os.path.join(os.path.dirname(os.path.realpath(__file__)), EXPERIMENT_SCRIPTS[exp])
        try:
            subprocess.run([str(sys.executable), str(script), "--config", str(self.config_path)], check=True, text=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Execution Error", f"Failed to execute experiment: {e}")
            # return
        self.block_idx += 1
        self._update_info()

if __name__ == "__main__":
    app = SessionGUI()
    app.mainloop()
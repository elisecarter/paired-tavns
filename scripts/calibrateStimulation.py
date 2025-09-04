import os
import random
import json
import time
import datetime
import keyboard  # if needed for additional functionality
import experiment_utils

class Calibrator:
    """
    A class to handle stimulation calibration.
    """
    def __init__(self,outdir, conditions,frequency=30):

        # Create the stimulation task object from experiment_utils
        stim_task = experiment_utils.initialize_stimulation_task('Dev1', frequency, 1)
        self.stim_task = stim_task
        self.outdir = outdir
        self.conditions = conditions
        self.results = {}

    def send_calibration_pulses(self):
        """
        Send pulse trains continuously until the user interrupts with CTRL+C.
        """
        print("Sending pulse trains. Press CTRL+C to stop.")
        try:
            while True:
                self.stim_task.start()
                self.stim_task.wait_until_done()
                self.stim_task.stop()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stim_task.stop()
            time.sleep(0.5)  # Give some time for the task to stop

    def send_single_pulse(self):
        """
        Sends a single pulse.
        """
        print("Sending single pulse.")
        self.stim_task.start()
        self.stim_task.wait_until_done()
        self.stim_task.stop()

    def calculate_current_values(self, threshold):
        """
        Given a perceptual threshold (in mA), calculate current values for predefined percentages.
        Returns a dict of values.
        """
        percentages = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
        values = {f"{int(p*100)}%": round(threshold * p, 3) for p in percentages}
        return values

    def run(self):
        """
        Run through the calibration process for each condition.
        Creates a data directory and saves a JSON file with the results.
        Returns the calibration results as a dictionary.
        """
        # Create data directory for saving calibration settings.
        for condition in self.conditions:
            print(f"\n--- Calibration for {condition} condition ---")
            input("Increase the stimulation current until perceptual threshold is reached. Press ENTER to begin.")
            self.send_calibration_pulses()
            
            # Get perceptual threshold from the user.
            while True:
                raw_input_value = input(f"Enter the perceptual threshold for {condition} (in mA): ")
                try:
                    threshold = float(raw_input_value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
            
            values = self.calculate_current_values(threshold)
            print(f"Calculated currents for {condition}: {values}")

            # Loop through calculated current values
            status = {}
            rating = {}
            current_keys = list(values.keys())
            for idx, (percentage, value) in enumerate(values.items()):
                # Skip if current value exceeds safe limit (5mA)
                if value > 5:
                    print(f"Skipping {percentage} ({value} mA) as it exceeds 5 mA.")
                    status[percentage] = "exceeds limit"
                    break
                # Skip if rating for previous percentage > 7
                if idx > 0:
                    try:
                        prev_rating = float(rating.get(current_keys[idx-1], 0))
                    except ValueError:
                        prev_rating = 0
                    if prev_rating > 7:
                        print(f"Skipping {percentage} ({value} mA) as previous rating was > 7.")
                        status[percentage] = "skipped due to previous rating > 7"
                        break
                try:
                    input(f"Set current amplitude to {percentage} ({value} mA). Press ENTER to send pulse. (Press CTRL+C to skip remaining pulses)")
                    self.send_single_pulse()
                    status[percentage] = "sent"
                    # Ensure that the rating is a valid number.
                    while True:
                        rating_input = input("Enter perceived sensation (0-10): ")
                        try:
                            numeric_rating = float(rating_input)
                            rating[percentage] = numeric_rating
                            break
                        except ValueError:
                            print("Invalid rating. Please enter a valid number.")
                    time.sleep(1)
                except (KeyboardInterrupt, EOFError):
                    print("\nCalibration interrupted by user.")
                    status[percentage] = "skipped"
                    # Mark remaining percentages as skipped.
                    for rem in current_keys[idx+1:]:
                        status[rem] = "skipped"
                    time.sleep(0.5)
                    break

            self.results[condition] = {
                "threshold": threshold,
                "calculated_currents": values,
                "perceived rating": rating
            }

            # Prompt to switch electrode position if not on final condition.
            if condition != self.conditions[-1]:
                input("Reset current and switch electrode position. Press ENTER to continue.")

        print("Calibration complete.")
        return self.results

# Example usage:
# if __name__ == "__main__":
#     # Example configuration
#     class Config:
#         outdir = "calibration_data"
#         conditions = ["sham", "taVNS"]

#     cfg = Config()
    
#     # Ensure output directory exists
#     data_dir = cfg.outdir
#     os.makedirs(data_dir, exist_ok=True)

#     # Run the calibration
#     calibrator = Calibrator(cfg)
#     results = calibrator.run()
#     print("Calibration completed. Results:", results)
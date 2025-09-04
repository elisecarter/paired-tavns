from psychopy import visual, core
import datetime
from experiment_utils import Experiment
import argparse, json


def run_plrt(exp, config):
    # Window setup
    win = visual.Window(fullscr=True, color='grey', units='norm',screen=1)
    fixation = visual.TextStim(win, text='+', color='black', height=0.2, pos=(0, 0))
    core.wait(5)

    for trial in range(config['num_trials']):

        # Start fixation
        fixation.autoDraw = True
        win.flip()
        trialClock = core.MonotonicClock()  # track exact fixation time
        exp.log_event("fixation_start")
        stimStarted = False
        flashStarted = False
        flashEnded = False

        elapsed = 0
        while elapsed < (config['baseline_dur'] + config['stim_dur']):
            elapsed = trialClock.getTime()  # how many seconds have passed in fixation

            # Start stimulation train and light stimulus at t=baseline_dur
            if (elapsed >= config['baseline_dur']) and (not stimStarted):
                win.color = 'white'
                if exp.trigger_stim: 
                    exp.stim_task.start()
                    exp.log_event(f"stim_{exp.condition}") # LSL event
                win.flip()
                win.flip()  # Ensure the window is refreshed after setting the color
                exp.log_event("flash_start")
                stimStarted = True
                flashStarted = True
 
            # End flash after 0.5s
            if (flashStarted and not flashEnded) and (elapsed >= config['baseline_dur'] + config['light_stim_dur']):
                win.color = 'grey'
                win.flip()
                win.flip()  # Ensure the window is refreshed after setting the color
                exp.log_event("flash_end")
                flashEnded = True

        if exp.trigger_stim: exp.stim_task.stop()
    win.close()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stroop Task")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)

    # Define experiment configuration
    config.update({
        "datetime": datetime.datetime.today(),
        "experiment": "PLRT",
        "send_trigger_codes": False,  # send trigger codes to DAQ
        "trigger_codes": '',  # Trigger codes for different events
        "num_trials": 10,   # Number of trials in the experiment
        "baseline_dur": 2.0,    # Duration of fixation before stim/light stimulus
        "poststim_dur": 0,     # Duration of fixation after response
        "stim_dur": 3,    # stimulation train duration
        "light_stim_dur": 0.5  # duration of light flash
    })

    if config['condition'] == "practice":
        config.update({
            "num_trials": 2,  # Fewer trials for practice
            "trigger_stim": False
        })

    # Initialize the experiment with configuration; note that you might also pass condition to Experiment.
    exp = Experiment(config)
    exp.setup_data_streams()
    exp.setup_stimulation_trigger()
    
    # Run the task...
    print("Starting light response task...")
    try:
        run_plrt(exp, config)
    except Exception as e:
        print("An error occurred during the task:", e)
        exp.stop_threads()
        exp.cleanup()
        raise
    exp.save_data()
    exp.stop_threads()
    exp.cleanup()
    print("PLRT completed.")



from psychopy import visual, core, event
import random, datetime
import argparse, json
import csv, os
import sys
from experiment_utils import Experiment


def run_stroop_task(exp, config):
    """
    Stroop Color-Word Task
    """
    # Create data directories & paths
    date_str = exp.datetime.strftime("%Y%m%d")
    datetime_str = exp.datetime.strftime("%Y%m%d-%H%M")
    stroopDir = os.path.join("Data", exp.ID, date_str, datetime_str)
    if not os.path.exists(stroopDir):
        os.makedirs(stroopDir)
    stroopPath = os.path.join(stroopDir, f"{datetime_str}_stroopTrials.csv")
    
    # Window setup
    win = visual.Window(fullscr=True, color='grey', units='norm',screen=1)
    fixation = visual.TextStim(win, text='+', color='black', height=0.2, pos=(0, 0))
    stimulus = visual.TextStim(win, text='', color='black', height=0.25, pos=(0, 0))
    buttons = {
        'red': visual.TextBox2(win, text='RED', color='black', letterHeight=0.08, pos=(-0.6, -0.5), size=(0.3, 0.3),borderColor='black',alignment='center'),
        'green': visual.TextBox2(win, text='GREEN', color='black', letterHeight=0.08, pos=(-0.20, -0.5), size=(0.3, 0.3),borderColor='black',alignment='center'),
        'blue': visual.TextBox2(win, text='BLUE', color='black', letterHeight=0.08, pos=(0.20, -0.5), size=(0.3, 0.3),borderColor='black',alignment='center'),
        'yellow': visual.TextBox2(win, text='YELLOW', color='black', letterHeight=0.08, pos=(0.6, -0.5), size=(0.3, 0.3),borderColor='black', alignment='center'),
    }
    # Calculate number of trials per condition
    num_incongruent_trials = int(config['num_trials'] * config['ratio_incongruent'])
    num_congruent_trials = config['num_trials'] - num_incongruent_trials
 
    # Generate trials with equal number of congruent and incongruent stimuli
    words = ['RED', 'GREEN', 'BLUE', "YELLOW"]
    colors = ['red', 'green', 'blue','yellow']
    key_mapping = {'red': ['r', '4','a'], 
                   'green': ['g','3','s'], 
                   'blue': ['b', '2', 'd'], 
                   'yellow': ['y', '1', 'f']}
    # Build a reverse mapping to quickly lookup the color from a key press
    reverse_mapping = {k: color for color, keys in key_mapping.items() for k in keys}
    all_keys = [key for keys in key_mapping.values() for key in keys]
    trials = []
    for _ in range(num_congruent_trials): # Generate congruent trials
        word = random.choice(words)
        color = word.lower()
        trials.append({'word': word, 'color': color, 'congruent': True})
    for _ in range(num_incongruent_trials): # Generate incongruent trials
        word = random.choice(words)
        color = random.choice([c for c in colors if c != word.lower()])
        trials.append({'word': word, 'color': color, 'congruent': False})
    random.shuffle(trials)

    # Show instructions
    instructions = visual.TextStim(
        win, text=(
            "In this task, you will see words displayed in different colors.\n\n"
            "Your job is to press the key corresponding to the COLOR of the word, ignoring what the word says.\n\n"
            "Press any key to start."
        ),
        color="white", height=0.1, pos=(0, 0)
    )
    instructions.draw()
    win.flip()
    event.waitKeys() 
    core.wait(1)

    # Open a CSV file to save data
    with open(stroopPath, 'w', newline='') as csvfile:
        fieldnames = ['trial_number', 'trial_type', 'stim_word', 'stim_color', 'response_side',  'rt', 'correct', 'score']
        stroopWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        stroopWriter.writeheader()

        score = 0
        for idx, trial in enumerate(trials):
            # Start fixation
            fixation.autoDraw = True
            win.flip()
            if exp.send_trigger_codes: exp.send_trigger('fixation_start') # send trigger for congr
            exp.log_event("fixation_start")
            #stimStarted = False
            core.wait(config['baseline_dur'])  # Wait for baseline duration
            
            # Show stimulus
            stimulus.setText(trial['word'])
            stimulus.setColor(trial['color'])
            stimulus.draw()
            for button in buttons.values():
                button.draw()
            fixation.autoDraw = False
            win.flip()
            trialClock = core.MonotonicClock()  # track exact stimulus (cue) time
            # if exp.trigger_stim: exp.stim_task.start()
            
            # Push trigger code to DAQ
            if exp.send_trigger_codes:
                trigger_code = 'cue_congruent' if trial['congruent'] else 'cue_incongruent'
                exp.send_trigger(trigger_code)

        
            event_label = 'cue_congruent' if trial['congruent'] else 'cue_incongruent'
            exp.log_event(event_label)

            # Wait for response
            keys = event.waitKeys(keyList=all_keys,timeStamped=trialClock) # type: ignore
            rt = keys[0][1]  # Get the response time from the key press # type: ignore

            # Get the color associated with the key pressed
            response = reverse_mapping.get(keys[0][0]) if keys else None
            correct = response == trial['color']
            
            if correct:
                if exp.send_trigger_codes: exp.send_trigger('response_correct') # send trigger code for response
                exp.log_event('response_correct') # LSL event
                if exp.trigger_stim: 
                    exp.stim_task.start()
                    exp.log_event(f"stim_{exp.condition}") # LSL event
                exp.correct_sound.stop()  # Stop the sound if it was playing
                exp.correct_sound.play()
                score += 1
                #buttons[response].borderColor = 'green'  # Change button color to green for correct response
            else:
                if exp.send_trigger_codes: exp.send_trigger('response_incorrect') # send trigger code for response
                exp.log_event('response_incorrect') # LSL event
                exp.incorrect_sound.stop()  # Stop the sound if it was playing
                exp.incorrect_sound.play()
                score -= 1
                #buttons[response].borderColor = 'red'  # Change button color to red for incorrect response
                    
            fixation.draw()
            win.flip()
            response_side = 'left' if response in ['red', 'green'] else 'right' if response in ['blue', 'yellow'] else 'unknown'
            # Save data
            stroopWriter.writerow({
                'trial_number': idx + 1,
                'trial_type': trial['congruent'],
                'stim_word': trial['word'],
                'stim_color': trial['color'],
                'response_side': response_side,
                'rt': rt,
                'correct': correct,
                'score': score
            })

            
            # Wait until post-stimulus duration 
            remaining_time =  (rt + config['poststim_dur']) - trialClock.getTime()
            if remaining_time > 0:
                core.wait(remaining_time)
            if exp.trigger_stim:
                exp.stim_task.stop()
            
            
        visual.TextStim(win, text=f"End of task.\n\n Score: {score}", color="white", height=0.1, pos=(0, 0)).draw()
        win.flip()
        core.wait(3)  # Wait for 3 seconds before closing the window
        win.close()

        

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stroop Task")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    config = load_config(args.config)

    # trigger_codes = {
#     'fixation_start': 1, # 2 on MSI
#     'cue_congruent': 2, # 4 on MSI
#     'cue_incongruent': 3, # 6 on MSI
#     'response_correct': 4, # 8 on MSI
#     'response_incorrect': 5, # 10 on MSI
#     'stim_on': 6, #12 on MSI               ###############ADD taVNS/sham####
#     'stim_off': 7,

    # Define experiment configuration
    config.update({
        "datetime": datetime.datetime.today(),
        "experiment": "SCWT",
        "send_trigger_codes": False,  # send trigger codes to DAQ
        "trigger_codes": '',
        "num_trials": 20,   # Number of trials in the experiment
        "ratio_incongruent": 0.5,   # Ratio of incongruent trials to total trials
        "baseline_dur": 2.0,    # Duration of fixation before Stroop cue
        "poststim_dur": 3.0,     # Duration of fixation after response
        "stim_dur": 0.5,    # stimulation train duration
    })

    if config['condition'] == "practice":
        config.update({
            "trigger_stim": False,
            "num_trials": 10,
        })

    # Initialize the experiment with configuration
    exp = Experiment(config)
    exp.setup_data_streams()
    exp.setup_stimulation_trigger()
    
    # Run the task...
    print("Starting stroop task...")
    try:
        run_stroop_task(exp, config)
    except Exception as e:
        print("An error occurred during the task:", e)
        exp.stop_threads()
        exp.cleanup()
        raise
    exp.save_data()
    exp.stop_threads()
    exp.cleanup()
    print("Stroop task completed.")

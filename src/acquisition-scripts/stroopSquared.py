from psychopy import visual, core, event
import random, datetime
from experiment_utils import Experiment
import csv, os
import argparse, json


def run_stroop_task(exp, config):
    """
    Stroop Squared Task developed by the Engle Lab
    Burgoyne, A. P., Tsukahara, J. S., Mashburn, C. A., Pak, R., & Engle, R. W. (2023). 
    Nature and measurement of attention control. 
    Journal of Experimental Psychology: General, 52(8), 2369–2402. 
    https://doi.org/10.1037/xge0001408
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
    fixation = visual.TextStim(win, text='+', color='black', height=0.2)
    stimulus = visual.TextStim(win, text='', color='black', height=0.25)

    buttons = {
        'left': visual.TextBox2(win, text='', color='black', letterHeight=0.1, pos=(-0.25, -0.5), size=(0.4, 0.4),borderColor='black',alignment='center'),
        'right': visual.TextBox2(win, text='', color='black', letterHeight=0.1, pos=(0.25, -0.5), size=(0.4, 0.4),borderColor='black',alignment='center'),
    }

    # Generate Stroop Squared trials (4 types, equal count)
    # words = ['RED', 'GREEN', 'BLUE', 'YELLOW']
    # colors = ['red', 'green', 'blue', 'yellow']
    words = ['RED', 'BLUE']
    colors = ['red', 'blue']
    key_mapping = {'left': ['l','a','s'], 
                   'right': ['r','d','f'] }
    # Build a reverse mapping to quickly lookup the color from a key press
    reverse_mapping = {k: side for side, keys in key_mapping.items() for k in keys}
    all_keys = [key for keys in key_mapping.values() for key in keys]
    
    num_types = round(config['num_trials'] * config['ratio_incongruent'])  # Number of trials per type
    trials = []
    # Fully congruent (target and response options are congruent
    for _ in range(num_types):
        word = random.choice(words) 
        color = word.lower() # cue color is congruent 
        options = ["", ""]
        options[0] = color.upper() # options (words) must contain target color
        options[1] = random.choice([w for w in words if w != options[0]]) # choose a different color
        random.shuffle(options) # shuffle options to randomize left/right
        left_opt, right_opt = options
        left_color = left_opt.lower() # options are congruent
        right_color = right_opt.lower()
        correct_response = 'left' if left_opt == color.upper() else 'right'
        trials.append({
            'word': word, 'color': color,
            'left': left_opt, 'left_color': left_color,
            'right': right_opt, 'right_color': right_color,
            'correct_response': correct_response,
            'trial_type': 'cueCongruent_responsesCongruent'
        })
    
    # Fully incongruent (target and response options are incongruent)
    for _ in range(num_types):
        word = random.choice(words) 
        color = random.choice([c for c in colors if c != word.lower()]) # target is incongruent
        options = ["", ""]
        options[0] = color.upper() # options (words) must contain target color
        options[1] = random.choice([w for w in words if w != options[0]]) # choose a different color
        random.shuffle(options) # shuffle options to randomize left/right
        left_opt, right_opt = options
        left_color = random.choice([c for c in colors if c != left_opt.lower()]) # options are incongruent
        right_color = random.choice([c for c in colors if (c != right_opt.lower() and c != left_color)])
        correct_response = 'left' if left_opt == color.upper() else 'right'
        trials.append({
            'word': word, 'color': color,
            'left': left_opt, 'left_color': left_color,
            'right': right_opt, 'right_color': right_color,
            'correct_response': correct_response,
            'trial_type': 'cueIncongruent_responsesIncongruent'
        })
    
    # Target incongruent, responses congruent 
    for _ in range(num_types):
        word = random.choice(words) 
        color = random.choice([c for c in colors if c != word.lower()]) # target is incongruent
        options = ["", ""]
        options[0] = color.upper() # options (words) must contain target color
        options[1] = random.choice([w for w in words if w != options[0]]) # choose a different color
        random.shuffle(options) # shuffle options to randomize left/right
        left_opt, right_opt = options
        left_color = left_opt.lower() # options are congruent
        right_color =  right_opt.lower()
        correct_response = 'left' if left_opt == color.upper() else 'right'
        trials.append({
            'word': word, 'color': color,
            'left': left_opt, 'left_color': left_color,
            'right': right_opt, 'right_color': right_color,
            'correct_response': correct_response,
            'trial_type': 'cueIncongruent_responsesCongruent'
        })
        
    # Target congruent, responses incongruent
    for _ in range(num_types):
        word = random.choice(words) 
        color = word.lower() # target is congruent
        options = ["", ""]
        options[0] = color.upper() # options (words) must contain target color
        options[1] = random.choice([w for w in words if w != options[0]]) # choose a different color
        random.shuffle(options) # shuffle options to randomize left/right
        left_opt, right_opt = options
        left_color = random.choice([c for c in colors if c != left_opt.lower()]) # options are incongruent
        right_color = random.choice([c for c in colors if (c != right_opt.lower() and c != left_color)])
        correct_response = 'left' if left_opt == color.upper() else 'right'
        trials.append({
            'word': word, 'color': color,
            'left': left_opt, 'left_color': left_color,
            'right': right_opt, 'right_color': right_color,
            'correct_response': correct_response,
            'trial_type': 'cueCongruent_responsesIncongruent'
        })
    
    # Randomize the order of trials
    random.shuffle(trials)

    # Show instructions
    instructions = visual.TextStim(
        win, text=(
            "In this task, you will see a colored word at the top and two response options below.\n\n"
            "You will chose the word whose MEANING matches the COLOR of the top word.\n\n"
            "Press any key to begin."
        ),
        color="white", height=0.1, pos=(0, 0)
    )
    instructions.draw()
    win.flip()
    event.waitKeys(keyList=all_keys) 
    core.wait(1) # wait for 1 second before starting the task
    
    # Open a CSV file to save data
    with open(stroopPath, 'w', newline='') as csvfile:
        fieldnames = ['trial_number', 'trial_type', 'stim_word', 'stim_color', 'response_side',  'rt', 'correct', 'score',
                      'left_word','left_color','right_word', 'right_color']

        stroopWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        stroopWriter.writeheader()
        
        score = 0
        clock = core.MonotonicClock() 
        start_time = clock.getTime()
        while (clock.getTime() - start_time) < 90: # run for 90 seconds
            for idx, trial in enumerate(trials):     
            
                # Show stimulus
                stimulus.setText(trial['word'])
                stimulus.setColor(trial['color'])
                stimulus.draw()
                for side in ['left', 'right']:
                    buttons[side].setText(trial[side])
                    buttons[side].setColor(trial[f'{side}_color'])
                    buttons[side].draw()
                win.flip()
                trialClock = core.MonotonicClock()  # track exact stimulus time
                
                # # Push trigger code to DAQ
                # if exp.send_trigger_codes:
                #     trigger_code = 'cue_congruent' if trial['congruent'] else 'cue_incongruent'
                #     exp.send_trigger(trigger_code)

                
                event_label = trial['trial_type']
                exp.log_event(event_label)

                # Wait for response
                keys = event.waitKeys(keyList=all_keys,timeStamped=trialClock) # type: ignore
                if clock.getTime() - start_time > 90:
                    break
                rt = keys[0][1]  # type: ignore      # Get the response time from the key press

                # Get the color associated with the key pressed
                response = reverse_mapping.get(keys[0][0]) if keys else None
                correct = response == trial['correct_response']
                
                if correct:
                    #if exp.send_trigger_codes: exp.send_trigger('response_correct') # send trigger code for response
                    exp.log_event('response_correct') # LSL event
                    if exp.trigger_stim: 
                        exp.stim_task.start()
                        exp.log_event(f"stim_{exp.condition}")
                    exp.correct_sound.stop()  # Stop the sound if it was playing
                    exp.correct_sound.play()
                    score += 1
                    #buttons[response].borderColor = 'green'  # Change button color to green for correct response
                else:
                    #if exp.send_trigger_codes: exp.send_trigger('response_incorrect') # send trigger code for response
                    exp.log_event('response_incorrect') # LSL event
                    exp.incorrect_sound.stop()  # Stop the sound if it was playing
                    exp.incorrect_sound.play()
                    score -= 1
                    #buttons[response].borderColor = 'red'  # Change button color to red for incorrect response
                        
                fixation.draw()
                win.flip()

                # Save data
                stroopWriter.writerow({
                    'trial_number': idx + 1,
                    'trial_type': trial['trial_type'],
                    'stim_word': trial['word'],
                    'stim_color': trial['color'],
                    'response_side': response,
                    'rt': rt,
                    'correct': correct,
                    'score': score,
                    'left_word': trial['left'],
                    'left_color': trial['left_color'],
                    'right_word': trial['right'],
                    'right_color': trial['right_color'],
                })

                
                # Wait until post-stimulus duration 
                remaining_time =  (rt + config['poststim_dur']) - trialClock.getTime()
                if remaining_time > 0:
                    core.wait(remaining_time)
                if exp.trigger_stim:
                    exp.stim_task.stop()


        visual.TextStim(win, text=f"End of task.\n\n Score: {score} ", color="white", height=0.1, pos=(0, 0)).draw()
        win.flip()
        core.wait(3)  # Wait for 2 seconds before closing the window
        # Close the window
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

    # Define experiment configuration
    config.update({
        "datetime": datetime.datetime.today(),
        "experiment": "StroopSquared",
        "send_trigger_codes": False,  # send trigger codes to DAQ
        "trigger_codes": '',
        "num_trials": 100,   # Number of trials in the experiment
        "ratio_incongruent": 0.25,   # Ratio of incongruent trials to total trials
        "baseline_dur": 0,    # Duration of fixation before Stroop cue
        "poststim_dur": 1,     # Duration of fixation after response
        "stim_dur": 0.5,    # stimulation train duration
    })
    if config['condition'] == "practice":
            config.update({
                "trigger_stim": False
        })

    # Initialize the experiment with configuration
    exp = Experiment(config)
    exp.setup_data_streams()
    exp.setup_stimulation_trigger()

    # Run the task...
    print("Starting stroop squared task...")
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
    print("Stroop squared task completed.")
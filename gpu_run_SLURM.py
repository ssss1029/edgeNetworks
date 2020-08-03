# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = " "

    SLURM_HEADER = "srun --pty -p gpu_jsteinhardt -w shaodwfax -c 10 --gres=gpu:1g"

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        "rdc_adv_stepSize8_epsilon_8_numSteps1_lr1e-8_advtargetBernoulli" : "python3 train_RCF_adv.py --resume=/data/sauravkadavath/RCFcheckpoint_epoch12.pth --print_freq=1000 --tmp=checkpoints/rcf_adv_stepSize8_epsilon_8_numSteps1_lr1e-8 --num-steps=1 --step-size=8 --epsilon=8 --lr=1e-8 --adv-target=bernoulli,
        
        "rdc_adv_stepSize8_epsilon_8_numSteps1_lr1e-8_advtargetOpposite" : "python3 train_RCF_adv.py --resume=/data/sauravkadavath/RCFcheckpoint_epoch12.pth --print_freq=1000 --tmp=checkpoints/rcf_adv_stepSize8_epsilon_8_numSteps1_lr1e-8 --num-steps=1 --step-size=8 --epsilon=8 --lr=1e-8 --adv-target=opposite",
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 1


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value


for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Running \"{0}\" with SLURM".format(command))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{0} {1}' C-m".format(
        Config.SLURM_HEADER, command
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)

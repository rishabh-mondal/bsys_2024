#!/bin/bash

# Define the path to the Python script
SCRIPT_PATH="/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/bsys_2024/florance_fintune.py"

# Define the output log file
LOG_FILE="/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/bsys_2024/florance_finetune.log"

# Run the Python script using nohup and redirect output to the log file
nohup python $SCRIPT_PATH > $LOG_FILE 2>&1 &

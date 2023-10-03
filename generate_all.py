#!/usr/bin/env python3

import subprocess

subprocess.run(["python", "generate.py", "--data_path train_dataset_16_09_23.txt"])
subprocess.run(["python", "generate.py", "--data_path valid_dataset_16_09_23.txt"])
subprocess.run(["python", "generate.py", "--data_path test_dataset_16_09_23.txt"])

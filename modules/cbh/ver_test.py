import os
from subprocess import call
import conda.cli
import pandas as pd
import fileinput
import json
import config as config
import shutil
import sys

tmp_dir = config.PROJECT_DIR/"tmp/"
if os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
os.mkdir(tmp_dir)
os.chdir(tmp_dir)
jsonf = tmp_dir / "requirements.json"

sys.stdout = open(jsonf, 'w') # Something here that provides a write method.
reqs = conda.cli.main('conda', 'list', '--json', '-e') # calls to print, ie import module1
sys.stdout.close() # close the redirected stdout
sys.stdout = sys.__stdout__ # restore the previous stdout.
reqs = pd.read_json(jsonf)
# print(reqs)
os.chdir(config.PROJECT_DIR)
shutil.rmtree(tmp_dir)

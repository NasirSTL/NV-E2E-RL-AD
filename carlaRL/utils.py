import sys, os
import pandas as pd

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

# save training log
def save_log(save_dir, logs):
    df = pd.DataFrame(logs)
    train_log_save_filename = os.path.join(save_dir, f'log.csv')
    df.to_csv(train_log_save_filename, index=False, encoding='utf-8')
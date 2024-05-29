import os
import yaml
import math
import torch
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime


plt.rcParams["font.family"] = "Times New Roman"
FIG_DPI=100

def print_log(text, file_name = 'log.txt', 
              ends_with = '\n', display = False):
    '''
    Prints output to the log file.
    
    text        : string or List               
                        Output text

    file_name   : string
                        Target log file

    ends_with   : string
                        Ending condition for print func.

    display     : Bool
                        Wheter print to screen or not.
    '''
    
    if display:
        print(text, end = ends_with)

    with open(file_name, "a") as text_file:
        print(text, end = ends_with, file = text_file)
        

def init_log(args):
    """
    Creates an unique output folder with given tag,
    inits log.txt, results/ and backups .py files in code/
    """
    # output folders
    start_time = datetime.now()
    output_dir = f"logs/{args.training_tag}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(output_dir+"code/")
    os.makedirs(output_dir+"results/")
    print("Output dir-->", output_dir)
    
    # backup
    for f in os.listdir("./"):
        if f.endswith(".py"):
            shutil.copyfile(f, output_dir+"/code/"+f)
    save_args(args, output_dir+"results/args.yaml")


    # log txt
    log_txt = os.path.join(output_dir, "log.txt")
    print_log('\n\tTraining "{}" started at: {} \n'.format(args.training_tag, 
                                                           start_time.strftime('%d-%m-%Y %H:%M:%S')), 
                                                           log_txt)
    print_log('GPU: {}'.format(torch.cuda.get_device_name()), 
              log_txt)
    print_log('Properties: {}\n'.format(torch.cuda.get_device_properties("cuda")), 
              log_txt)

    return output_dir, log_txt


def remove_tailzeros(arr):
    """Remove trailing zeros from a 1D NumPy array."""
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input should be a numpy array")
    if arr.ndim != 1:
        raise ValueError("Input array should be 1D")
    
    # Find the index of the last non-zero element
    last_nonzero = np.nonzero(arr)[0]
    
    if len(last_nonzero) == 0:
        return np.array([])
    
    return arr[:last_nonzero[-1] + 1]


def time2sec(time_str, return_ms = False):
    '''
    Converts hh:mm:ss or hh:mm:ss,ms
    to seconds or miliseconds

    sec             : String
                        Time
    return_ms       : Bool
                        Wheter return miliseconds
    '''
    if len(time_str.split(",")) == 2:
        hms = time_str.split(",")[0]
        hours = int(hms.split(":")[0])
        minutes = int(hms.split(":")[1])
        seconds = int(hms.split(":")[2])
        miliseconds = int(time_str.split(",")[1])
    else:
        hours = int(time_str.split(":")[0])
        minutes = int(time_str.split(":")[1])
        seconds = int(time_str.split(":")[2])
        miliseconds = 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds

    if return_ms:
        return total_seconds * 1000 + miliseconds
    else:
        return total_seconds


def listdir(path, ending = None):
    '''Returns dir with full path'''
    if ending == None:
        return sorted([os.path.join(path, f) for f in os.listdir(path)])
    else:
        return sorted([os.path.join(path, f) for f in os.listdir(path) 
                       if f.endswith(ending)])
    

def save_args(args, filename):
    # Convert the argparse Namespace to a dictionary
    params = vars(args)
    
    # Save the dictionary to a YAML file
    with open(filename, 'w') as file:
        yaml.dump(params, file, default_flow_style=False)
    

def plot_error(error_train, error_valid, output_path):
    error_train = remove_tailzeros(error_train.cpu().detach().numpy())
    error_valid = remove_tailzeros(error_valid.cpu().detach().numpy())
       
    plt.figure(dpi=FIG_DPI, constrained_layout=True)
    plt.plot(error_train, color='#084c61', label='Train')
    plt.plot(error_valid, color='#ffc857', label='Valid')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.savefig(output_path+'results/error_function.jpg')
    plt.close('all')
    
    
def plot_rsd(predicted_rsd, true_rsd, output_path, op_name):
    predicted_rsd = predicted_rsd.cpu().detach().numpy()*10000
    true_rsd = true_rsd.cpu().detach().numpy()*10000
    running_mean = np.convolve(predicted_rsd, np.ones(5)/5, mode='valid')    
    
    plt.figure(dpi=FIG_DPI, constrained_layout=True)
    plt.plot(predicted_rsd, color='#db3a34', label='Predicted')
    plt.plot(true_rsd, color='#084c61', label='True')
    plt.plot(running_mean, color='#ffb703', label='Mean')
    plt.xlabel('Sentences', fontsize=16)
    plt.ylabel('RSD (s)', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.savefig(output_path+f'results/{op_name}.jpg')
    plt.close('all')
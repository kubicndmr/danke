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

def prefix(id, name = '', buffer = 3):
    '''
    Creates prefix padded with zeros, e.g., name001
    
    id              : int
                        Number to be padded

    name            : string
                        Prefix

    buffer          : int
                        Length of numbers including zeros
    '''
    return name + str(id).zfill(buffer)


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
    if isinstance(arr, np.ndarray):    
        last_nonzero = np.nonzero(arr)[0]
        if len(last_nonzero) == 0:
            return np.array([])
        else:
            return arr[:last_nonzero[-1] + 1]
    """Remove trailing zeros from a torch array."""
    if isinstance(arr, torch.Tensor):    
        last_nonzero = torch.nonzero(arr, as_tuple=True)[0]
        if len(last_nonzero) == 0:
            return torch.tensor([])
        else:
            return arr[:(last_nonzero[-1].item() + 1)]


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
    

def data_split(data_path, log_txt='log.txt', split=0.8, synthetic_data_path=None):
    # read path
    ops = listdir(data_path, '.pkl')

    # shuffle
    random.seed(1881)
    random.shuffle(ops)
    
    # compute number of ops in testset
    if len(ops)%2 == 0:
        testset_size = math.ceil(len(ops)*(1-split))
    else:
        testset_size = math.ceil(len(ops-1)*(1-split))
    
    # split
    trainset = ops[:-2*testset_size]
    validset = ops[-2*testset_size:-testset_size]
    testset = ops[-testset_size:]
    
    #synthetic
    if synthetic_data_path != None:
        trainset = trainset + listdir(synthetic_data_path, '.pkl')
    
    # log
    print_log("\tTrainset [{}] Data Channels\t: {}".format(len(trainset),
                                                    trainset), log_txt)
    print_log("\tValidset [{}] Data Channels\t: {}".format(len(validset), 
                                                    validset), log_txt)
    print_log("\tTestset [{}] Data Channels\t: {}".format(len(testset), 
                                                    testset), log_txt)
        
    return [trainset,validset, testset]


def plot_error(error_train, error_valid, output_path):
    error_train = remove_tailzeros(error_train.cpu().detach().numpy())
    error_valid = remove_tailzeros(error_valid.cpu().detach().numpy())
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
        
    plt.figure(dpi=FIG_DPI, constrained_layout=True)
    plt.plot(error_train, color='#084c61', linewidth=2, label='Train')
    plt.plot(error_valid, color='#a6382e', linewidth=2, label='Valid')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.savefig(output_path+'results/error_function.jpg')
    plt.close('all')
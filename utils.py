import os
import math
import yaml
import time
import torch
import shutil
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import KFold

plt.rcParams["font.family"] = "Times New Roman"
FIG_DPI = 100


def prefix(id, name='', buffer=3):
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


def print_log(text, file_name='log.txt',
              ends_with='\n', display=False):
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
        print(text, end=ends_with)

    with open(file_name, "a") as text_file:
        print(text, end=ends_with, file=text_file)


def init_log(output_dir, backup=True):
    start_time = datetime.now()
    os.makedirs(output_dir)
    
    if backup:
        os.makedirs(output_dir+"code/")
        os.makedirs(output_dir+"results/")
        #os.makedirs(output_dir+"results/ribbons/")
    print("Output dir-->", output_dir)

    if backup:
        for f in os.listdir("./"):
            if f.endswith(".py"):
                shutil.copyfile(f, output_dir+"/code/"+f)

    # log txt
    log_txt = os.path.join(output_dir, "log.txt")
    print_log('\n\tTraining started at: {} \n'.format(
        start_time.strftime('%d-%m-%Y %H:%M:%S')),
        log_txt)
    print_log('\n---{ Hardware }---', log_txt)
    print_log('GPU: {}'.format(torch.cuda.get_device_name()),
              log_txt)
    print_log('Properties: {}\n'.format(torch.cuda.get_device_properties("cuda")),
              log_txt)

    return log_txt


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


def time2sec(time_str, return_ms=False):
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


def listdir(path, ending=None):
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


def data_split(real_data_path: str, syn_data_path: str, 
               real_dataset_size: int, syn_dataset_size: int, 
               eval_ratio: float, n_splits: int, log_txt: str):
    '''
    Splits real and synthetic datasets into training and evaluation subsets.
    Also supports k-fold cross-validation for the real dataset and logs the split details.

    real_data_path    : string
                        Path to the directory containing the real dataset files.

    syn_data_path     : string
                        Path to the directory containing the synthetic dataset files.

    real_dataset_size : int
                        Number of real data samples to use. If -1, all available data is used.

    syn_dataset_size  : int
                        Number of synthetic data samples to use. If -1, all available data is used.

    eval_ratio        : float
                        Proportion of data to allocate to the evaluation set (e.g., 0.2 for 20%).

    n_splits          : int
                        Number of folds for k-fold cross-validation on the real dataset.

    log_txt           : string
                        Path to the log file where the split details will be recorded.
    '''

    #### Synthetic Data ####
    # Check Split ratio
    assert 0 < eval_ratio < 0.5, "Evaluation split ratio should be in [0, 0.5] range!" 

    # Synthetic Dataset (For Pretraining - Classic)
    syn_dataset = listdir(syn_data_path, '.pkl')        
    
    # Select all data if -1
    if syn_dataset_size == -1:
        syn_dataset_size = len(syn_dataset)
        
    # Split Sets
    if syn_dataset_size <= len(syn_dataset):
        syntrainset_size = math.ceil(syn_dataset_size*(1-eval_ratio))
        syn_evalset_size = math.ceil(syn_dataset_size*eval_ratio)
        syntrainset = syn_dataset[:syntrainset_size]
        syntestset = syn_dataset[syntrainset_size:syntrainset_size+syn_evalset_size]
    else:
        raise ValueError(f"[SYN] You asked {syn_dataset_size} OPs, but the '{syn_data_path}' has {len(syn_dataset)} OPs instead")
    
    #### Real Data ####
    # Check n_splits
    assert 1 < n_splits < 6, "Folds should be in [2, 5] range!" 
    
    # Real Dataset (For Finetuning - k-Fold)
    real_dataset = listdir(real_data_path, '.pkl')

    # Select all data if -1
    if real_dataset_size == -1:
        real_dataset_size = len(real_dataset)
    
    if real_dataset_size > len(real_dataset):
        raise ValueError(f"[Real] You asked {real_dataset_size} OPs, but the '{real_data_path}' has {len(real_dataset)} OPs instead")
    
    realtrainsets = []
    realtestsets = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train, test in kf.split(real_dataset[:real_dataset_size]):
        realtrainsets.append([real_dataset[i] for i in train])
        realtestsets.append([real_dataset[i] for i in test])
    
    #### Log ####
    print_log('\n---{ Synthetic Data }---', log_txt)
    print_log("Trainset [{}] OPs\t: {}".format(len(syntrainset),
                                                         syntrainset), log_txt)
    print_log("Testset [{}] OPs\t: {}".format(len(syntestset), 
                                                        syntestset), log_txt)
    
    print_log('\n---{ Real Data }---', log_txt)
    for i in range(n_splits):
        print_log("Fold {}| Trainset [{}] OPs\t: {}".format(i+1, len(realtrainsets[i]),
                                                                      realtrainsets[i]), log_txt)
        print_log("Fold {}| Testset [{}] OPs\t: {}".format(i+1, len(realtestsets[i]),
                                                                     realtestsets[i]), log_txt)
            
    return {'syntrainset': syntrainset,
            'syntestset': syntestset,
            'realtrainsets': realtrainsets,
            'realtestsets': realtestsets}


def plot_error(error_train, error_valid=None, output_path='', tag='', fig_dpi=100):
    """
    Plots the training and validation error over epochs.

    Parameters:
    - error_train (array-like): Training error, can be a numpy array or PyTorch tensor.
    - error_valid (array-like, optional): Validation error, can be a numpy array or PyTorch tensor. Default is None.
    - output_path (str): Directory path to save the plot. Default is an empty string.
    - tag (str): Tag to append to the saved file name. Default is an empty string.
    - fig_dpi (int): Dots per inch (DPI) for the figure. Default is 100.
    """
    def process_error(error):
        """Converts to numpy and removes trailing zeros if necessary."""
        if error is None:
            return None
        if not isinstance(error, np.ndarray):
            error = error.cpu().detach().numpy()  # Convert PyTorch tensor to numpy
        return remove_tailzeros(error)

    # Process input errors
    error_train = process_error(error_train)
    error_valid = process_error(error_valid)

    # Plot settings
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    # Create figure
    plt.figure(dpi=fig_dpi, constrained_layout=True)

    # Check if error arrays are 1D or 2D
    if error_train.ndim == 1:
        # Plot single curve for each
        plt.plot(error_train, color='#084c61', linewidth=2, label='Train')
        if error_valid is not None:
            plt.plot(error_valid, color='#a6382e', linewidth=2, label='Valid')
        plt.xticks(ticks=np.linspace(0, len(error_train), 5),
                   labels=np.linspace(1, len(error_train)+1, 5))
    else:
        # Set zeros to NaN for ignoring them in the plot
        error_train[error_train == 0] = np.nan
        if error_valid is not None:
            error_valid[error_valid == 0] = np.nan
        # Plot each repetition's curve
        for i in range(error_train.shape[1]):
            # Lower alpha for better overlap visibility
            plt.plot(error_train[:, i], color='#084c61',
                     linewidth=2, alpha=0.7)
            if error_valid is not None:
                plt.plot(error_valid[:, i], color='#a6382e',
                         linewidth=2, alpha=0.7)
        # Add labels only once for the legend
        plt.plot([], [], color='#084c61', linewidth=2, label='Train')
        if error_valid is not None:
            plt.plot([], [], color='#a6382e', linewidth=2, label='Valid')

        plt.xticks(ticks=np.linspace(0, error_train.shape[0], 5),
                   labels=np.linspace(1, error_train.shape[0], 5))

    # Labels and legend
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)

    # Save plot
    plt.savefig(f'{output_path}results/{tag}error_function.jpg')
    plt.close('all')


def plot_ribbon(data, title, output_path, repeat=16):
    ''' Plots color ribbon with legend

    data        : np.array [1xN]
                    Data to plot

    title       : str
                    Title and save name of the figure, e.g. OP name

    output_path : str
                    path to save.

    repeat      : int
                    Vertical width of the ribbon 
    '''
    save_path = os.path.join(output_path, title+'.jpg')

    # Labels
    phases = ['Preperation', 'Puncture', 'GuideWire', 'CathPlacement',
              'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing', 'Transition']

    # Check data type
    assert type(data) == type(
        np.zeros([1, 1])), "Input data should be a numpy array"

    # Adapt shape
    data = np.expand_dims(data, 0)

    # Repeat for thickness
    data = np.repeat(data, repeats=repeat, axis=0)
    formatter = matplotlib.ticker.FuncFormatter(lambda s,
                                                x: time.strftime('%M:%S', time.gmtime(s // 60)))
    xtick_pos = np.linspace(0, data.shape[1], data.shape[1] // 350)

    # Cmap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 9))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'DMap', color_list, 9)

    # Plot
    plt.figure(dpi=FIG_DPI)
    plt.matshow(data, cmap=disc_cmap, vmin=0, vmax=8)
    plt.grid(False)
    plt.yticks([])
    plt.clim(-0.5, 8.5)
    cbar = plt.colorbar(ticks=range(len(phases)))
    cbar.ax.set_yticks(np.arange(len(phases)), labels=phases)
    plt.xticks(xtick_pos, fontsize=18)
    plt.gca().xaxis.tick_bottom()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel('Time (HH:MM)')
    plt.title(title, fontsize=20, pad=10)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def phase_weights(dataset, save_path=None):
    if len(dataset) == 0:
        return torch.ones(8, dtype=torch.float32)
    else:
        phase_count = np.zeros(8, dtype=int)
        for d in dataset:
            phase_count += d.dataset.phase_count()

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, phase_count)
        per_cls_weights = (1.0 - beta) / effective_num
        per_cls_weights = per_cls_weights / \
            np.sum(per_cls_weights) * len(phase_count)

        # plot
        if save_path != None:
            percentage_count = np.zeros((len(dataset), 8))
            for i, d in enumerate(dataset):
                count = d.dataset.phase_count()
                percentage_count[i, :] = count / np.sum(count)

            def_cmap = plt.cm.get_cmap('tab10')
            color_list = def_cmap(np.linspace(0, 1, 9))

            plt.figure(dpi=FIG_DPI)
            boxplots = plt.boxplot(percentage_count,
                                patch_artist=True,
                                medianprops=dict(color='black')
                                )
            for patch, color in zip(boxplots['boxes'], color_list):
                patch.set_facecolor(color)

            plt.xticks(ticks=range(1, 9), labels=range(8), fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Surgical Phases', fontsize=18)
            plt.ylabel('Percentage (%)', fontsize=18)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close('all')

        return torch.FloatTensor(per_cls_weights)


def plot_metrics_over_runs(metrics_dict, metric_keys, output_path):
    for metric_name in metric_keys:
        plt.figure(dpi=100, constrained_layout=True)

        # Iterate over each run for the specific metric and plot it
        for run_idx in range(metrics_dict[metric_name].shape[0]):
            # Lower alpha for overlapping effect
            plt.plot(remove_tailzeros(
                metrics_dict[metric_name][run_idx, :]), alpha=0.7, color='black')

        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel(metric_name, fontsize=14)
        plt.title(
            f'{metric_name} over {metrics_dict[metric_name].shape[0]} runs')

        # Save the plot
        plt.savefig(f'{output_path}results/{metric_name}.jpg')
        plt.close()


def first_zero_index(arr, max_epoch):
    zero_indices = np.where(arr == 0)[0]
    return zero_indices[0] if zero_indices.size > 0 else max_epoch


def compute_run_means(results_array):
    non_zero_sum = np.sum(results_array, axis=0)
    non_zero_counts = np.count_nonzero(results_array, axis=0)
    non_zero_counts[non_zero_counts == 0] = 1
    return non_zero_sum / non_zero_counts

def plot_confusion_matrix(confusion_matrices, output_path):
    
    sum_cm = np.zeros((8, 8))
    for cm in confusion_matrices:
        sum_cm += cm
    
    # Normalize
    sum_cm = sum_cm / sum_cm.sum(axis=1, keepdims=True)
    
    # Plot confusion matrix
    plt.figure(dpi=600, constrained_layout=True)
    plt.imshow(sum_cm, cmap='Blues')
    plt.xticks(ticks=range(8), labels=range(8))
    plt.yticks(ticks=range(8), labels=range(8))
    plt.ylabel('True Phases', fontsize=20)
    plt.xlabel('Predicted Phases', fontsize=20)
    plt.savefig(f'{output_path}results/confusion_matrix.png', bbox_inches='tight')
    plt.close()
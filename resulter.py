import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Result summarizer")

parser.add_argument('-f', '--experiment_folder', type=str)
parser.add_argument('-t', '--experiment_tag', type=str)
args = parser.parse_args()

experiment_folder = args.experiment_folder
experiment_list = os.listdir(experiment_folder)

real_acc = np.zeros(len(experiment_list))
real_f1 = np.zeros(len(experiment_list))
real_jac = np.zeros(len(experiment_list))
syn_acc = np.zeros(len(experiment_list))
syn_f1 = np.zeros(len(experiment_list))
syn_jac = np.zeros(len(experiment_list))

for i, experiment in enumerate(experiment_list):
    with open(os.path.join(experiment_folder, experiment, 'log.txt'), 'r') as log_file:
        for line in log_file:
            if line.startswith('[validation-Real]\tMax Acc'):
                real_acc[i] = line[30:37]
            elif line.startswith('[validation-Real]\tMax F1'):
                real_f1[i] = line[29:36]
            elif line.startswith('[validation-Real]\tMax Jaccard'):
                real_jac[i] = line[34:41]
            elif line.startswith('[validation-Syn]\tMax Acc'):
                syn_acc[i] = line[29:37]
            elif line.startswith('[validation-Syn]\tMax F1'):
                syn_f1[i] = line[28:35]
            elif line.startswith('[validation-Syn]\tMax Jaccard'):
                syn_jac[i] = line[33:40]

print('--{'+args.experiment_tag+'}--')
print('\t   [Mean]\t  [Std]\n'+'-'*35)
print(f'[r] acc\t: {np.mean(real_acc):.5f}\t {np.std(real_acc):.5f}')
print(f'[r] f1\t: {np.mean(real_f1):.5f}\t {np.std(real_f1):.5f}')
print(f'[r] jac\t: {np.mean(real_jac):.5f}\t {np.std(real_jac):.5f}')
print(f'[s] acc\t: {np.mean(syn_acc):.5f}\t {np.std(syn_acc):.5f}')
print(f'[s] f1\t: {np.mean(syn_f1):.5f}\t {np.std(syn_f1):.5f}')
print(f'[s] jac\t: {np.mean(syn_jac):.5f}\t {np.std(syn_jac):.5f}')

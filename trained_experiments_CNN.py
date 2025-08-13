import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from IPython.display import display, Latex
from collections import defaultdict
import pandas as pd

from torch.distributions.multivariate_normal import MultivariateNormal
from datetime import datetime as dt

from networks import *
from network_derivatives import *
from utils import *
from plotting_func import *
from load_data import *

import os
from tqdm import tqdm
import yaml

from pathlib import Path
import requests
import pickle
import gzip
import scipy

import mnist_reader

def cal_hess_information(x_train, y_train, loss_func, network, calc_H, method_cond_num='naive', device='cpu', **kwargs):
     

    if calc_H:

        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'num_channels':[],
                                'kernel_size':[],
                                'depth':[],
                                'activ_f':[],
                                'lr':[],
                                'epoch':[],
                                'H_cond':[],
                                'H_o_cond':[],
                                'lam_abs_min_H':[],
                                'lam_abs_max_H':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'mean_diff_H_H_o':[],
                                'max_diff_H_H_o':[],
                                'std_diff_H_H_o':[],
                                'H_rank':[],
                                'H_o_rank':[],
                                'H_spectrum':[],
                                'H_spectrum_abs':[],
                                'H_o_spectrum':[],
                                },dtype=object)
                                
        _H_cond, _H_o_cond, _lam_abs_min_H, _lam_abs_max_H, _lam_abs_min_H_o, _lam_abs_max_H_o, _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o, _H_rank, _H_o_rank, _H_spectrum, _H_spectrum_abs, _H_o_spectrum = calc_condition_num(network,
                                                            x_train,y_train,
                                                            loss_func,
                                                            device,
                                                            calc_H,
                                                            method_cond_num)
        
#         hessian_information.append([network.input_dim, network.output_dim, 
#                                                          network.num_channels, network.depth, network.activation_func, 
#                                                          kwargs['epoch'],
#                                                         _H_cond, _H_o_cond, 
#                                                         _lam_abs_min_H, _lam_abs_max_H, 
#                                                         _lam_abs_min_H_o, _lam_abs_max_H_o,
#                                                         _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
#                                                         _H_rank, _H_o_rank,
#                                                         _H_spectrum, _H_o_spectrum], ignore_index=True)
        

       
     
        hessian_information.loc[len(hessian_information)] = [network.input_dim, network.output_dim, 
                                                         network.num_channels, network.kernel_size, network.depth, network.activation_func, 
                                                         kwargs['lr'],
                                                         kwargs['epoch'],
                                                        _H_cond, _H_o_cond, 
                                                        _lam_abs_min_H, _lam_abs_max_H, 
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _mean_diff_H_H_o, _max_diff_H_H_o, _std_diff_H_H_o,
                                                        _H_rank, _H_o_rank, _H_spectrum, _H_spectrum_abs, _H_o_spectrum]
    
    else:
       hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'num_channels':[],
                                'kernel_size':[],
                                'depth':[],
                                'activ_f':[],
                                'lr':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })
       _H_o_cond, _lam_abs_min_H_o, _lam_abs_max_H_o, _H_o_rank, _H_o_spectrum = calc_condition_num(network,
                                                            x_train,y_train,
                                                            loss_func,
                                                            device,
                                                            calc_H,
                                                            method_cond_num)
       hessian_information.loc[len(hessian_information)] = [network.input_dim, network.output_dim, 
                                                         network.num_channels, network.kernel_size, network.depth, network.activation_func,kwargs['lr'],
                                                        kwargs['epoch'],
                                                        _H_o_cond,
                                                        _lam_abs_min_H_o, _lam_abs_max_H_o,
                                                        _H_o_rank]
    
    
    # print('Epoch: 0 \t loss= %10.3e' %loss_func(network(x_train), y_train).detach())
      
    return hessian_information

    
def eval_network_configurations(networks, x_train, y_train, loss_func, calc_H, method_cond_num='naive', device='cpu', **kwargs):
    

    if calc_H:
        hessian_information = pd.DataFrame({'input_dim':[],
                                            'output_dim':[],
                                            'num_channels':[],
                                            'kernel_size':[],
                                            'depth':[],
                                            'activ_f':[],
                                            'lr':[],
                                            'epoch':[],
                                            'H_cond':[],
                                            'H_o_cond':[],
                                            'lam_abs_min_H':[],
                                            'lam_abs_max_H':[],
                                            'lam_abs_min_H_o':[],
                                            'lam_abs_max_H_o':[],
                                            'mean_diff_H_H_o':[],
                                            'max_diff_H_H_o':[],
                                            'std_diff_H_H_o':[],
                                            'H_rank':[],
                                            'H_o_rank':[]
                                            })
    else:
        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'num_channels':[],
                                'kernel_size':[],
                                'depth':[],
                                'activ_f':[],
                                'lr':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })


#     print('Initializing Networks...')

#     print(np.linalg.cond(x_train.T@x_train))
#     print(x_train.shape)

    for ind, network in enumerate(networks):

        print('Network configuration: d=%d, k=%d, m=%d, L=%d' % (networks[0].input_dim, networks[0].output_dim, network.num_channels, network.depth))
        
#         print(x_train.device)

        start_t = time.time()
    
        _hessian_information = cal_hess_information(x_train, y_train, 
                                                    loss_func, network, 
                                                    calc_H=calc_H,
                                                    method_cond_num=method_cond_num,
                                                    device=device, epoch=kwargs['epochs'][ind], lr=kwargs['lr'])

        hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)


        hessian_information['H_o_cond'] = hessian_information['H_o_cond'].astype(float)
        if calc_H:
            hessian_information['mean_diff_H_H_o'] = hessian_information['mean_diff_H_H_o'].astype(float)
            hessian_information['max_diff_H_H_o'] = hessian_information['max_diff_H_H_o'].astype(float)
            hessian_information['std_diff_H_H_o'] = hessian_information['std_diff_H_H_o'].astype(float)

        print(f'time passed: {time.time()-start_t}')
    return hessian_information

def main(project_name, experiment_name, config):

    device = config['device']
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Running experiment %s for project %s ' %(experiment_name, project_name))
    
    # Define DataFrames to log information about training process and Hessian information (condition number, eigenvalues etc.)
    dataset = config['dataset']
    if dataset == 'gaussian':
        x_train, y_train, _, _ = load_run(dataset_path)

        x_train = torch.tensor(x_train)
    elif dataset == 'mnist' or dataset == 'fashion' :
        x_train, y_train, _, _ = load_mnist(dataset, config['datapoints'], config['downsample_factor'],config['whiten'], device)
    elif dataset == 'cifar-10' :
        x_train, y_train, _, _ = load_cifar10(config['datapoints'], config['grayscale'], config['flatten'], config['whiten'], device)
    else:
        raise ValueError('Unknown dataset')
    
    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                    'network':[],
                                    'cond_cov_xx':[],
                                    'input_dim':[],
                                    'output_dim':[],
                                    'num_channels':[],
                                    'kernel_size':[],
                                    'depth':[],
                                    'activ_f':[],
                                    'lr':[],
                                    'epoch':[],
                                    'type':[],
                                    'value':[],
                                    'H_o_spectrum':[]
                                    },dtype=object)
    
    
    if config['calc_H']:
        hessian_information = pd.DataFrame({'input_dim':[],
                                            'output_dim':[],
                                            'num_channels':[],
                                            'kernel_size':[],
                                            'depth':[],
                                            'activ_f':[],
                                            'lr':[],
                                            'epoch':[],
                                            'H_cond':[],
                                            'H_o_cond':[],
                                            'lam_abs_min_H':[],
                                            'lam_abs_max_H':[],
                                            'lam_abs_min_H_o':[],
                                            'lam_abs_max_H_o':[],
                                            'mean_diff_H_H_o':[],
                                            'max_diff_H_H_o':[],
                                            'std_diff_H_H_o':[],
                                            'H_rank':[],
                                            'H_o_rank':[],
                                            'H_spectrum':[],
                                            'H_o_spectrum':[]
                                            })
    else:
        hessian_information = pd.DataFrame({ 
                                'input_dim':[],
                                'output_dim':[],
                                'num_channels':[],
                                'kernel_size':[],
                                'depth':[],
                                'activ_f':[],
                                'lr':[],
                                'epoch':[],
                                'H_o_cond':[],
                                'lam_abs_min_H_o':[],
                                'lam_abs_max_H_o':[],
                                'H_o_rank':[]
                                })

    time_start = datetime.datetime.now()
        
    
    dataset_path = config['dataset_path']
    dataset = config['dataset']

#     epochs = config['epochs']
    epoch_max = config['epoch_max']
    every_epoch = config['every_epoch']
    if epoch_max > 1:
        epochs = np.append(np.arange(0,epoch_max,every_epoch),epoch_max-1)
    else:
        epochs = np.zeros(1)
         
    # first check if all networks exist before starting to calculate!!
    
    for l in config['L']:
        for lr in config['lr']:
            for init in range(config['inits']):
                
                lr = '%.6f' % float(lr)
                
                filename = f"{config['experiment_name']}_{config['dataset']}_NOTwhitened_init={init}_network_d=49_m={config['num_channel']}_kernel_sz={config['kernel_size']}_k=10_L={l}_relu_SGD_lr={lr}_BS={config['BS']}"

                for epoch in epochs:
                    filepath = config['path']+filename + '_epoch=%d' %epoch + '.pt' 

                    network = torch.load(filepath)
    
    print('All specified networks exist.')
    
    
    
    
#     for filename in config['filenames']:
    for l in config['L']:
        for lr in config['lr']:
            for init in range(config['inits']):
           
                lr = '%.6f' % float(lr)

                filename = f"{config['experiment_name']}_{config['dataset']}_NOTwhitened_init={init}_network_d=49_m={config['num_channel']}_kernel_sz={config['kernel_size']}_k=10_L={l}_relu_SGD_lr={lr}_BS={config['BS']}"


    #                 filename = f"{config['experiment_name']}_{config['dataset']}_NOTwhitened_init={init}_network_d=49_m={config['m']}_k=10_L={l}_relu_SGD_lr={lr}_BS={config['BS']}"


                Networks = [] # list of NN with different configurations
                for epoch in epochs:
                    filepath = config['path']+filename + '_epoch=%d' %epoch + '.pt' 

                    network = torch.load(filepath)
                    network = network.to(device)
                # network = torch.jit.load(config['path']+filename)
                # network.eval()
                    num_param = sum([len(param.flatten()) for param in network.parameters()]) 
                    print(filename)
                    print(f'Num parameters = {num_param}')
                    num_pos_param = sum([len(param[torch.abs(param)>0].flatten()) for param in network.parameters()])
                    print(f'Proportion of non-zero params = {num_pos_param/num_param}')

                    Networks.append(network)

                if config['loss_func'] == 'mse':
                    loss_func = F.mse_loss
                elif config['loss_func'] == 'crossentropy':
                    loss_func = nn.CrossEntropyLoss()


                if config['calc_hess_info'] == True:
                    _hessian_information  = eval_network_configurations(Networks, x_train, y_train, loss_func, calc_H=config['calc_H'], method_cond_num=config['method_cond_num'],device=device, epochs=epochs, lr=lr)


                if config['calc_hess_info'] == True:
                    hessian_information = pd.concat([hessian_information, _hessian_information],ignore_index=True)
                            
                


                print('Time passed: %.3f seconds' %(datetime.datetime.now()-time_start).total_seconds())         
    
    time_now = dt.now().isoformat()

    file_path = 'figures/' + config['project_name'] + '_' + config['experiment_name'] + '_' + time_now + '/'
    os.mkdir(file_path)

    if config['log_yaml_file']  == True:
        yaml_file_name = file_path + 'config_' + time_now + '.yaml'
        with open(yaml_file_name, 'w') as file:
            yaml.dump(config, file)

    if config['calc_hess_info'] == True:
        hessian_information.to_pickle(f"pandas_dataframes_new/full_hessian_information_{config['experiment_name']}_{config['dataset']}_NOTwhitened_m={config['num_channel']}_kernelsz={config['kernel_size']}_L={config['L']}_lr={config['lr']}_BS={config['BS']}.pkl")

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    plt.rcParams["figure.figsize"] = (4,4)

        
#     super_config = {}
#     config_path = 'super_config.yaml'
#     with open(config_path, 'r') as file:
#         super_config.update(yaml.safe_load(file))
        
# #     config_files = super_config['config_files']
    config_files = ['config_trained_experiments_CNN.yaml']
    
    for base_config_path in config_files:
        # load in YAML configuration
        config = {}

#         base_config_path = 'config_trained_experiments.yaml'
        with open(base_config_path, 'r') as file:
            config.update(yaml.safe_load(file))

        # start training with name and config 
        main(config['project_name'], config['experiment_name'], config)


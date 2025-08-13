import sys
sys.path.append("cifar10/")

from network_derivatives import *
from cifar10.tnt_solver import *
from load_data import *
import pandas as pd


def calc_H_O(network,x_train,device):
    '''
    Calculates the outer product Hessian of the loss via the Jacobian of the network w.r.t. the parameters
    Returns: the Jacobian times its transposed averaged over all datapoints, and its non-zero spectrum
    
    network: Neural network
    x: input samples
    '''
    
    print('Calculating H_O...')
    
        
    start_tt = time.time()
    i = 0
    
    bs=64
    end = x_train.shape[0] // bs
    for ind in range(end+1):
        
    #     print(compute_jacobian(model_orig,b[0].unsqueeze(0)).shape)
        
        x = x_train[ind*bs:(ind+1)*bs,:].to(torch.device(device))   
        
        print(x.device, x.shape)
        
        dim_0 = x.shape[0]*10
        
#         print(x.device, x.shape)
        start_t = time.time()
    #         print(x[i,:].shape)
        if i==0:
            jacob = compute_jacobian(network,x).detach().view(dim_0,-1).to(torch.device('cpu'))
        else:
            jacob = torch.concatenate((jacob, 
                                       compute_jacobian(network,x).detach().view(dim_0,-1).to(torch.device('cpu'))), axis=0)
        
        print('i:', ind, 'time=', time.time() - start_t)
        
        
        i+=1
        
    print(jacob.shape)
    jjT = jacob @ jacob.T
    
    print('jjT.shape=',jjT.shape)
    if jjT.shape[0] > 15000:
        print('Something is probably wrong!!')
        return 0 
        

    print('total time:', time.time() - start_tt)
    
    
#     jac_jac_T_rank = torch.linalg.matrix_rank(jac_jac_T, atol=1e-7/jac_jac_T.shape[0])
    # jac_jac_T_rank = torch.linalg.matrix_rank(jac_jac_T, atol=1e-7/jac_jac_T.shape[0])
    try:
        jac_jac_T_spectrum = torch.linalg.eigvalsh(jjT) #[-jac_jac_T_rank:]
    except:
        eps = 1e-19
        jac_jac_T_spectrum = torch.linalg.eigvalsh(jjT + eps*torch.eye(jjT.shape[0], device=torch.device(device)))
        
    jac_T_jac_spectrum, _ = torch.sort(torch.abs(jac_jac_T_spectrum))

#     jac_jac_T = jac_jac_T
    
    print('Finished calculating H_O and spectrum ...')
    
    
    return jjT.detach().cpu(), jac_jac_T_spectrum.detach().cpu()





data = CIFAR10Data(train_split=0.8)
train_itr = data.get_train_loader(batch_size=64)

pruned_weights = [0.0, 0.2] #, 0.2, 0.4, 0.6, 0.8
epochs = [0,40,80,120,160,199]

device = 'cuda:6'
n = 1000
grayscale = False
flatten = True
whiten = False

x_train, y_train, _, _ = load_cifar10(n, grayscale, flatten, whiten, device)

print('x_train shape: ', x_train.shape)

for p_w in pruned_weights:
    
    outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                    'num_samples':[],
                                    'network':[],
                                    'pruned_weights':[],
                                    'epoch':[],
                                    'H_o_cond':[],
                                    'H_o_spectrum':[],
                                    'H_o_rank':[],
                                    'lambda_max_H_o':[],
                                    'lambda_min_H_o':[]
                                    },dtype=object)
    
    for epoch in epochs:
        filepath = f'trained_models/training_n=1000_default_init_cifar-10_NOTwhitened_init=0_network_d=3072_m=128_k=10_L=3_relu_SGD_lr=0.050000_BS=64p_w={p_w}_epoch={epoch}.pt'
        
        print(filepath)
        
        network = torch.load(filepath)
        network = network.to(device)
        
        start_t = time.time()
        
        jjT, jac_jac_T_spectrum = calc_H_O(network,x_train,device)
        
        
        
        
        H_o_rank = torch.linalg.matrix_rank(jjT, atol=1e-7)
        lambda_max_H_o = jac_jac_T_spectrum[-1]
        lambda_min_H_o = jac_jac_T_spectrum[-H_o_rank]
        H_o_cond = lambda_max_H_o/lambda_min_H_o
        
        print('time to calculate:', time.time()-start_t)
        
        
        outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = ['Cifar10', n,
                                                                                   'FcNN',
                                                                                   p_w,
                                                                                   epoch,
                                                                                   H_o_cond, 
                                                                                   jac_jac_T_spectrum,
                                                                                   H_o_rank,
                                                                                   lambda_max_H_o,
                                                                                   lambda_min_H_o]
        

        
    
        outer_prod_hessian_information.to_pickle(f"panda_dataframes/outer_prod_hessian_information_FcNN_pruned_weights={p_w}_{epoch}.pkl")


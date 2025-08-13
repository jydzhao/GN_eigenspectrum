import sys
sys.path.append("cifar10/")

from network_derivatives import *
import pandas as pd

#@title Prepare Data 📊
# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
from vit import *


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        torch.manual_seed(6666)
        torch.cuda.manual_seed(6666)
        
        indices = torch.randperm(len(trainset))[:train_sample_size]
        trainset = torch.utils.data.Subset(trainset, indices)



    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        torch.manual_seed(3141)
        torch.cuda.manual_seed(3141)
        
        indices = torch.randperm(len(testset))[:test_sample_size]
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def calc_H_O(network,train_itr,device):
    '''
    Calculates the outer product Hessian of the loss via the Jacobian of the network w.r.t. the parameters
    Returns: the Jacobian times its transposed averaged over all datapoints, and its non-zero spectrum
    
    network: Neural network
    x: input samples
    '''
    
    print('Calculating H_O...')
    
        
    start_tt = time.time()
    i = 0

    for b in train_itr:
    #     print(compute_jacobian(model_orig,b[0].unsqueeze(0)).shape)
        dim_0 = b[0].shape[0]*10

        b = b[0].to(torch.device(device))
        print(b.device, b.shape)
        start_t = time.time()
    #         print(x[i,:].shape)
        if i==0:
            jacob = compute_jacobian(network,b).detach().view(dim_0,-1)
        else:
            jacob = torch.concatenate((jacob, 
                                       compute_jacobian(network,b).detach().view(dim_0,-1)), axis=0)
        print('i:', time.time() - start_t)
        # print(i,jacob.shape)

        i+=1
        
    jjT = jacob @ jacob.T
    
    print('jjT.shape=',jjT.shape)
    if jjT.shape[0] > 15000:
        print('Something is probably wrong!!')
        

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



config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

inits = 3
device = 'cuda:0'
pruned_weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] #, 
epochs = [20,40,60,80,100,120,140,160,180,'final']

trainloader, testloader, _ = prepare_data(batch_size=64, train_sample_size=1000, test_sample_size=1000)

for init in range(inits):
    for p_w in pruned_weights:
        
        outer_prod_hessian_information = pd.DataFrame({'dataset':[],
                                        'num_samples':[],
                                        'network':[],
                                        'init':[],
                                        'pruned_weights':[],
                                        'epoch':[],
                                        'H_o_cond':[],
                                        'H_o_spectrum':[],
                                        'H_o_rank':[],
                                        'lambda_max_H_o':[],
                                        'lambda_min_H_o':[]
                                        },dtype=object)
        
        for epoch in epochs:
            filepath = f'experiments/vit_p_w={p_w}/model_vit_p_w={p_w}_init_{init}_epoch={epoch}.pt'
            
            print(filepath)
            
            network = ViTForClassfication(config)
            network.load_state_dict(torch.load(filepath))
            network.eval()
            network = network.to(device)
            
            start_t = time.time()
            
            jjT, jac_jac_T_spectrum = calc_H_O(network,trainloader,device)
   
            
            H_o_rank = torch.linalg.matrix_rank(jjT, atol=1e-7)
            lambda_max_H_o = jac_jac_T_spectrum[-1]
            lambda_min_H_o = jac_jac_T_spectrum[-H_o_rank]
            H_o_cond = lambda_max_H_o/lambda_min_H_o
            
            print('time to calculate:', time.time()-start_t)
            
            
            outer_prod_hessian_information.loc[len(outer_prod_hessian_information)] = ['Cifar-10', 
                                                                                       1000,
                                                                                       'ViT',
                                                                                       init,
                                                                                       p_w,
                                                                                       epoch,
                                                                                       H_o_cond, 
                                                                                       jac_jac_T_spectrum,
                                                                                       H_o_rank,
                                                                                       lambda_max_H_o,
                                                                                       lambda_min_H_o]

        
            outer_prod_hessian_information.to_pickle(f"panda_dataframes/outer_prod_hessian_information_ViT_init={init}_pruned_weights={p_w}_{epoch}.pkl")


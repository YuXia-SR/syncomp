import syncomp.models.process_GQ as pce
import syncomp.models.autoencoder as ae
import syncomp.models.diffusion as diff
import syncomp.models.TabDDPMdiff as TabDiff
import pandas as pd
import time
import logging

def train_autodiff(
    train_df: pd.DataFrame,
    # Auto-encoder hyper-parameters 
    threshold = 0.01, # Threshold for mixed-type variables
    n_epochs = 10000, #@param {'type':'integer'}
    eps = 1e-5, #@param {type:"number"}
    weight_decay = 1e-6, #@param {'type':'number'}
    maximum_learning_rate = 1e-2, #@param {'type':'number'}
    lr = 2e-4, #@param {'type':'number'}
    hidden_size = 250,
    num_layers = 3,
    batch_size = 50,
    # Diffusion hyper-parameters
    diff_n_epochs = 10000, #@param {'type':'integer'}
    sigma = 20,  #@param {'type':'integer'} 
    num_batches_per_epoch = 50, #@param {'type':'number'}
    T = 300,  #@param {'type':'integer'}
    device='cpu', #@param {'type':'string'},
    **kwargs
):  
    
    logging.info("Training Autoencoder")
    ds = ae.train_autoencoder(train_df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()
    
    logging.info("Training Diffusion Model")
    score = TabDiff.train_diffusion(latent_features, T, eps, sigma, lr, \
                    num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)
    N = latent_features.shape[0] 
    P = latent_features.shape[1]
    
    start_time = time.time()
    sample = diff.Euler_Maruyama_sampling(score, T, N, P, device)
    end_time = time.time()
    duration = end_time - start_time

    logging.info("Generating synthetic data")
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(train_df, gen_output, threshold)

    return syn_df, duration
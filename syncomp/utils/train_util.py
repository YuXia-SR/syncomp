import syncomp.models.process_GQ as pce
import syncomp.models.autoencoder as ae
import syncomp.models.diffusion as diff
import syncomp.models.TabDDPMdiff as TabDiff
from syncomp.models.ctab_gan_model.ctabgan import CTABGAN
from ctgan import CTGAN
import pandas as pd
import time
import logging
import tqdm

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
    
    sample = diff.Euler_Maruyama_sampling(score, T, N, P, device)

    logging.info("Generating synthetic data")
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(train_df, gen_output, threshold)

    return syn_df

def train_ctgan(
    train_df: pd.DataFrame,
    epochs: int=100,
    **kwargs
):
    # Names of the columns that are discrete
    discrete_columns = list(train_df.select_dtypes(include=['object', 'int']).columns)

    ctgan = CTGAN(epochs=epochs, **kwargs)
    ctgan.fit(train_df, discrete_columns)
    syn_df = ctgan.sample(len(train_df))

    return syn_df
    

def train_ctabgan(
    train_df: pd.DataFrame,
    class_dim=(256, 256, 256, 256),
    random_dim=100,
    num_channels=64,
    l2scale=1e-5,
    batch_size=500,
    epochs=150
):
    categorical_columns = train_df.select_dtypes(include=['object']).columns
    integer_columns = train_df.select_dtypes(include=['int64']).columns
    synthesizer =  CTABGAN(
        train_df=train_df,
        categorical_columns=categorical_columns,
        integer_columns=integer_columns,
        class_dim=class_dim,
        random_dim=random_dim,
        num_channels=num_channels,
        l2scale=l2scale,
        batch_size=batch_size,
        epochs=epochs
    )
    synthesizer.fit()
    syn_df = synthesizer.generate_samples(len(train_df))
    return syn_df
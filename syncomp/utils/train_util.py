from ctgan import CTGAN
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import syncomp.models.process_GQ as pce
import syncomp.models.autoencoder as ae
import syncomp.models.diffusion as diff
import syncomp.models.TabDDPMdiff as TabDiff
from syncomp.models.ctab_gan_model.ctabgan import CTABGAN
from syncomp.models.autogan import train_gan, generate_gan

def train_tabautodiff(
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
                    num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size, device)
    N = latent_features.shape[0] 
    P = latent_features.shape[1]
    
    sample = diff.Euler_Maruyama_sampling(score, T, N, P, device)

    logging.info("Generating synthetic data")
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(train_df, gen_output, threshold)

    return syn_df


def train_stasyautodiff(
    train_df: pd.DataFrame,
    # Auto-encoder hyper-parameters 
    threshold = 0.01, # Threshold for mixed-type variables
    n_epochs = 10000, #@param {'type':'integer'}
    weight_decay = 1e-6, #@param {'type':'number'}
    lr = 2e-4, #@param {'type':'number'}
    hidden_size = 250,
    num_layers = 3,
    batch_size = 50,
    # Diffusion hyper-parameters
    diff_n_epochs = 100, #@param {'type':'integer'}
    sigma = 20,  #@param {'type':'integer'} 
    num_batches_per_epoch = 50, #@param {'type':'number'}
    T = 300,  #@param {'type':'integer'}
    hidden_dims = (256, 512, 1024, 512, 256), #@param {type:"raw"}
    maximum_learning_rate = 1e-2, #@param {'type':'number'}
    eps = 1e-5, #@param {type:"number"}
    device='cpu', #@param {'type':'string'},
    **kwargs
):  
    
    logging.info("Training Autoencoder")
    ds = ae.train_autoencoder(train_df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()
    
    logging.info("Training Diffusion Model")
    score = diff.train_diffusion(latent_features, T, hidden_dims, latent_features.shape[1], eps, sigma, lr, \
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
    epochs: int=500,
    device: str='cpu',
    **kwargs
):
    # Names of the columns that are discrete
    discrete_columns = list(train_df.select_dtypes(include=['object', 'int']).columns)

    ctgan = CTGAN(epochs=epochs)
    ctgan.set_device(device)
    ctgan.fit(train_df, discrete_columns)
    syn_df = ctgan.sample(len(train_df))

    return syn_df
    

def train_ctabgan(
    train_df: pd.DataFrame,
    class_dim=(256, 256, 256, 256),
    random_dim=100,
    num_channels=64,
    l2scale=1e-5,
    batch_size=100,
    epochs=500,
    **kwargs
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

def prepare_latent_features(
        train_df: pd.DataFrame,
        category_transformer,
        numerical_transformer,
):
    logging.info("Prepare latent features")
    # Separate categorical and numerical columns using select_dtypes
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    train_df[categorical_cols] = train_df[categorical_cols].astype(str)
    numerical_cols = train_df.select_dtypes(exclude=['object']).columns.tolist()
    train_df[numerical_cols] = train_df[numerical_cols].astype(float)

    # Pipeline for preprocessing categorical columns
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True, )),  # Impute missing values (if any)
        ('onehot', category_transformer)  # One-hot encode categorical columns
    ])

    # Pipeline for preprocessing numerical columns
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),  # Impute missing values (if any)
        ('quantile',numerical_transformer)  # Transform numerical columns with quantile transformer
    ])

    categorical_cols_features = categorical_pipeline.fit_transform(train_df[categorical_cols]).toarray().astype('float32')
    numerical_cols_features = numerical_pipeline.fit_transform(train_df[numerical_cols]).astype('float32')
    latent_features = np.hstack([categorical_cols_features, numerical_cols_features])
    n_categorical_features = categorical_cols_features.shape[1]

    return latent_features, n_categorical_features, categorical_pipeline, numerical_pipeline, categorical_cols, numerical_cols

def convert_syn_df(
        sample,
        n_categorical_features,
        categorical_pipeline,
        numerical_pipeline,
        categorical_cols,
        numerical_cols
):
    logging.info("Convert latent features to synthetic data")
    categorical_df = categorical_pipeline[1].inverse_transform(sample[:, :n_categorical_features])
    numerical_df = numerical_pipeline[1].inverse_transform(sample[:, n_categorical_features:])
    syn_df = pd.concat([pd.DataFrame(categorical_df, columns=categorical_cols), pd.DataFrame(numerical_df, columns=numerical_cols)], axis=1)

    return syn_df

def train_tabddpm(
    train_df: pd.DataFrame,
    eps = 1e-5, #@param {type:"number"}
    weight_decay = 1e-6, #@param {'type':'number'}
    maximum_learning_rate = 1e-2, #@param {'type':'number'}
    lr = 2e-4, #@param {'type':'number'}
    batch_size = 50,
    diff_n_epochs = 10000, #@param {'type':'integer'}
    sigma = 20,  #@param {'type':'integer'} 
    num_batches_per_epoch = 50, #@param {'type':'number'}
    T = 300,  #@param {'type':'integer'}
    device='cpu', #@param {'type':'string'},
):
    (
        latent_features, 
        n_categorical_features, 
        categorical_pipeline, 
        numerical_pipeline, 
        categorical_cols, 
        numerical_cols
    ) =prepare_latent_features(train_df, OneHotEncoder(handle_unknown='ignore'), QuantileTransformer())

    logging.info("Training diffusion model")
    N, P = latent_features.shape
    score = TabDiff.train_diffusion(latent_features, T, eps, sigma, lr, \
                num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)
    sample = TabDiff.Euler_Maruyama_sampling(score, T, N, P, device)
    
    syn_df = convert_syn_df(
        sample, 
        n_categorical_features, 
        categorical_pipeline, 
        numerical_pipeline, 
        categorical_cols, 
        numerical_cols
    )
    return syn_df


def train_stasy(
    train_df: pd.DataFrame,
    weight_decay = 1e-6, #@param {'type':'number'}
    lr = 2e-4, #@param {'type':'number'}
    batch_size = 50,
    # Diffusion hyper-parameters
    diff_n_epochs = 100, #@param {'type':'integer'}
    sigma = 20,  #@param {'type':'integer'} 
    num_batches_per_epoch = 50, #@param {'type':'number'}
    T = 300,  #@param {'type':'integer'}
    hidden_dims = (256, 512, 1024, 512, 256), #@param {type:"raw"}
    maximum_learning_rate = 1e-2, #@param {'type':'number'}
    eps = 1e-5, #@param {type:"number"}
    device='cpu', #@param {'type':'string'},
    **kwargs
):  
    
    (
        latent_features, 
        n_categorical_features, 
        categorical_pipeline, 
        numerical_pipeline, 
        categorical_cols, 
        numerical_cols
    ) =prepare_latent_features(train_df, OneHotEncoder(handle_unknown='ignore'), MinMaxScaler())

    N, P = latent_features.shape
    logging.info("Training Diffusion Model")
    score = diff.train_diffusion(latent_features, T, hidden_dims, latent_features.shape[1], eps, sigma, lr, \
                        num_batches_per_epoch, maximum_learning_rate, weight_decay, diff_n_epochs, batch_size)
    N = latent_features.shape[0] 
    P = latent_features.shape[1]
    sample = diff.Euler_Maruyama_sampling(score, T, N, P, device)
    
    syn_df = convert_syn_df(
        sample, 
        n_categorical_features, 
        categorical_pipeline, 
        numerical_pipeline, 
        categorical_cols, 
        numerical_cols
    )
    return syn_df


def train_autogan(
    train_df: pd.DataFrame,
    # Auto-encoder hyper-parameters 
    threshold = 0.01, # Threshold for mixed-type variables
    n_epochs = 10000, #@param {'type':'integer'}
    weight_decay = 1e-6, #@param {'type':'number'}
    lr = 2e-4, #@param {'type':'number'}
    hidden_size = 250,
    num_layers = 3,
    batch_size = 50,
    # GAN hyper-parameters
    gan_lr = 0.0002,
    b1 = 0.5,
    b2 = 0.999,
    gan_n_epochs = 300,
    gan_batch_size = 1000,
    generator_layers = [256, 512, 1024],
    discriminator_layers = [1024, 512, 256],
    generator_neg_slope = 0.2,
    discriminator_neg_slope = 0.2,
    device='cpu', #@param {'type':'string'},
    **kwargs
):  
    logging.info("Training Autoencoder")
    ds = ae.train_autoencoder(train_df, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold)
    latent_features = ds[1].detach()
    
    logging.info("Training GAN Model")
    generator = train_gan(
        latent_features, gan_lr, b1, b2, gan_n_epochs, gan_batch_size,
        generator_layers, discriminator_layers, generator_neg_slope, discriminator_neg_slope, device
    )
    N = latent_features.shape[0]
    P = latent_features.shape[1]
    sample = generate_gan(generator, N, P, device)

    logging.info("Generating synthetic data")
    gen_output = ds[0](sample, ds[2], ds[3])
    syn_df = pce.convert_to_table(train_df, gen_output, threshold)

    return syn_df
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, input_dim, layer_dims, negative_slope=0.2):
        super(Generator, self).__init__()
        self.model = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
            self.model.add_module(f'layer{i}', nn.Linear(in_dim, out_dim))
            self.model.add_module(f'activation{i}', nn.LeakyReLU(negative_slope))
        self.model.add_module(f'layer{len(layer_dims)}', nn.Linear(layer_dims[-1], input_dim))
        self.model.add_module(f'activation{len(layer_dims)}', nn.Tanh())
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, layer_dims, negative_slope=0.2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip([input_dim] + layer_dims[:-1], layer_dims)):
            self.model.add_module(f'layer{i}', nn.Linear(in_dim, out_dim))
            self.model.add_module(f'activation{i}', nn.LeakyReLU(negative_slope))
        self.model.add_module(f'layer{len(layer_dims)}', nn.Linear(layer_dims[-1], 1))
        self.model.add_module(f'activation{len(layer_dims)}', nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)

class TensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, index):
        return self.data_tensor[index]

def train_gan(
    features,
    lr=0.0002,
    b1=0.5,
    b2=0.999,
    n_epochs=200,
    batch_size=64,
    generator_layers=[256, 512, 1024],
    discriminator_layers=[1024, 512, 256],
    generator_neg_slope=0.2,
    discriminator_neg_slope=0.2,
    device='cpu'
):
    input_dim = features.shape[1]
    generator = Generator(input_dim=input_dim, layer_dims=generator_layers, negative_slope=generator_neg_slope).to(device)
    discriminator = Discriminator(input_dim=input_dim, layer_dims=discriminator_layers, negative_slope=discriminator_neg_slope).to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    adversarial_loss = nn.BCELoss().to(device)

    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        for i, real in enumerate(dataloader):
            # Move data to the appropriate device
            real = real.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(real.size(0), 1, device=device)
            fake = torch.zeros(real.size(0), 1, device=device)
            
            # Configure input
            real = real.view(real.size(0), -1)
            
            # -----------------
            #  Train Generator
            # -----------------
            
            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(real.size(0), input_dim, device=device)
            
            # Generate a batch of images
            gen = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen), valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real), valid)
            fake_loss = adversarial_loss(discriminator(gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            batches_done = epoch * len(dataloader) + i
            
        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    return generator

def generate_gan(generator, n_samples, input_dim, device='cpu'):
    # Set the generator to evaluation mode
    generator.eval()
    
    # Generate random noise
    noise = torch.randn(n_samples, input_dim, device=device)
    
    # Generate fake images
    with torch.no_grad():
        fake = generator(noise)
    
    # Reshape and scale images
    fake = fake.view(fake.size(0), input_dim)
    
    return fake

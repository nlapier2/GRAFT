import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchdiffeq import odeint

class PositionalEmbedding(nn.Module):
    """Simple sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # Accept t as (B,) or (B,1); coerce to (B,)
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        elif t.dim() != 1:
            t = t.reshape(-1)

        device = t.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        phase = t[:, None] * freq[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

class FlowModel(nn.Module):
    """A simple MLP that predicts the velocity vector for flow matching."""
    def __init__(self, n_latent=128, n_hidden=512, time_emb_dim=32):
        super().__init__()
        
        # A layer to embed the scalar time t into a higher-dimensional vector
        self.time_embed = PositionalEmbedding(time_emb_dim)
        
        # The main network that predicts the velocity
        self.net = nn.Sequential(
            # The input is the latent vector z concatenated with the time embedding
            nn.Linear(n_latent + time_emb_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent) # The output is the velocity vector, so its size is n_latent
        )

    def forward(self, z, t):
        # 1. Embed the time scalar t
        t_emb = self.time_embed(t)
        
        # 2. Concatenate the latent state z with the time embedding
        zt = torch.cat([z, t_emb], dim=1)
        
        # 3. Predict the velocity using the MLP
        return self.net(zt)

class VAE(nn.Module):
    """A Variational Autoencoder implemented in vanilla PyTorch."""
    def __init__(self, n_genes, n_latent=128, n_hidden=512):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2) # Outputs mu and log_var
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes)
        )

    def reparameterize(self, mu, log_var):
        """
        Performs the reparameterization trick to allow for backpropagation.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        """
        # Encode
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

class VAEFlowModel(nn.Module):
    """A joint model containing both the VAE and the Flow Model."""
    def __init__(self, n_genes, n_latent=128, n_hidden=512):
        super().__init__()

        # --- VAE Components ---
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2) # Outputs mu and log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes)
        )
        
        # --- Flow Model Component ---
        self.flow_model = FlowModel(n_latent, n_hidden)

    def reparameterize(self, mu, log_var):
        """Performs the reparameterization trick to allow for backpropagation."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Defines the forward pass for the VAE component.
        This is used for reconstruction and encoding.
        """
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

# --- Loss Functions ---

def calculate_vae_loss(x_hat, x, mu, log_var, beta):
    """
    Calculates the VAE loss (reconstruction + KL divergence).
    """
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Normalize KL divergence by batch size
    kl_divergence /= x.size(0)
    return reconstruction_loss + beta * kl_divergence

def calculate_flow_matching_loss(flow_model, z1):
    """
    Calculates the flow matching loss from a Gaussian prior to the encoded data z1.
    """
    # 1. Sample z0 from a standard Gaussian (our prior distribution)
    z0 = torch.randn_like(z1)
    
    # 2. Sample a random time t for each sample in the batch
    t = torch.rand(z1.size(0), device=z1.device)
    
    # 3. Calculate the point on the path (z_t) and the target velocity (v_target)
    zt = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1
    v_target = z1 - z0
    
    # 4. Get the model's predicted velocity
    v_pred = flow_model(zt, t)
    
    # 5. Return the MSE loss
    return F.mse_loss(v_pred, v_target)

def calculate_composite_loss(vae_loss, flow_loss, gamma):
    """
    Combines the VAE and flow matching losses.
    """
    return vae_loss + gamma * flow_loss

# --- The Main "Autoencoding Flow" Model ---

class AutoFlowModel(nn.Module):
    """
    A joint model implementing the "Autoencoding Flow" objective.
    The forward pass performs a full generation from noise to a reconstructed cell.
    """
    def __init__(self, n_genes, n_latent=128, n_hidden=512):
        super().__init__()

        # VAE Components
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2) # Outputs mu and log_var
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes)
        )
        
        # Flow Model Component
        self.flow_model = FlowModel(n_latent, n_hidden)

    def reparameterize(self, mu, log_var):
        """Performs the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, device):
        """
        Defines the full end-to-end forward pass for the Autoencoding Flow.
        """
        # 1. Encode the real data to get the target latent vector
        mu_real, log_var_real = torch.chunk(self.encoder(x), 2, dim=-1)
        z_real = self.reparameterize(mu_real, log_var_real)

        # 2. Sample a corresponding noise vector from the prior
        z_noise = torch.randn_like(z_real)

        # 3. Define the ODE function for the solver, which expects inputs (t, z)
        def ode_func(t, z):
            # Our model expects a batch of t values, so we expand the scalar t
            t_batch = t.expand(z.size(0))
            return self.flow_model(z, t_batch)

        # 4. Integrate from t=0 to t=1 to generate the predicted latent vector
        t_span = torch.tensor([0.0, 1.0], device=device)
        # We only need the final state at t=1
        z_pred = odeint(ode_func, z_noise, t_span, method='dopri5', rtol=1e-5, atol=1e-5)[1]

        # 5. Decode the prediction to get the final "fake" cell for comparison
        x_fake = self.decoder(z_pred)

        # Return all necessary components for the loss calculation
        return x_fake, z_pred, z_real, mu_real, log_var_real

    def reconstruct(self, x):
        """
        Performs a simple VAE reconstruction (encode -> decode).
        This is used for validation, separate from the main forward pass.
        """
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat
    
    @property
    def latent_dim(self):
        return self.encoder[-1].out_features // 2

    def forward_vae(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# --- Loss Functions for the New Objective ---

def calculate_autoencoding_flow_loss(x_fake, x_real, z_pred, z_real, gamma):
    """
    Calculates the composite loss for the Autoencoding Flow model.
    """
    # 1. The final reconstruction loss in gene space
    reconstruction_loss = F.mse_loss(x_fake, x_real)
    
    # 2. The flow matching loss in latent space
    flow_loss = F.mse_loss(z_pred, z_real)
    
    # 3. Return the weighted composite loss
    return reconstruction_loss + gamma * flow_loss

class LinearAutoFlowModel(nn.Module):
    """
    A joint model implementing the "Autoencoding Flow" objective but with a
    simple LINEAR decoder. This is designed to force the latent space to
    learn the complex correlation structure of the data.
    """
    def __init__(self, n_genes, n_latent=128, n_hidden=512):
        super().__init__()

        # --- VAE Components ---
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2) # Outputs mu and log_var
        )
        
        # --- THE KEY ARCHITECTURAL CHANGE ---
        # The powerful MLP decoder is replaced with a single linear layer.
        self.decoder = nn.Linear(n_latent, n_genes)
        # --- END OF CHANGE ---
        
        # --- Flow Model Component (Unchanged) ---
        self.flow_model = FlowModel(n_latent, n_hidden)

    def reparameterize(self, mu, log_var):
        """Performs the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reconstruct(self, x):
        """
        Performs a simple VAE reconstruction (encode -> decode).
        Used for validation.
        """
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x, device):
        """
        Defines the full end-to-end forward pass for the Autoencoding Flow.
        This logic remains the same as the previous model.
        """
        # 1. Encode the real data to get the target latent vector
        mu_real, log_var_real = torch.chunk(self.encoder(x), 2, dim=-1)
        z_real = self.reparameterize(mu_real, log_var_real)

        # 2. Sample a corresponding noise vector from the prior
        z_noise = torch.randn_like(z_real)

        # 3. Define the ODE function for the solver
        def ode_func(t, z):
            t_batch = t.expand(z.size(0))
            return self.flow_model(z, t_batch)

        # 4. Integrate from t=0 to t=1 to generate the predicted latent vector
        t_span = torch.tensor([0.0, 1.0], device=device)
        z_pred = odeint(ode_func, z_noise, t_span, method='dopri5', rtol=1e-5, atol=1e-5)[1]

        # 5. Decode the prediction to get the final "fake" cell
        x_fake = self.decoder(z_pred)

        # Return all necessary components for the loss calculation
        return x_fake, z_pred, z_real, mu_real, log_var_real
    
    @property
    def latent_dim(self):
        return self.encoder[-1].out_features // 2

    def forward_vae(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

def _encode_mu(model, x):
    mu, _ = torch.chunk(model.encoder(x), 2, dim=-1)
    return mu

def latent_cycle_consistency(model, data_batch):
    with torch.no_grad():
        mu_real = _encode_mu(model, data_batch)    # target detached
    x_rec = model.decoder(mu_real)
    mu_rec = _encode_mu(model, x_rec)
    return torch.mean((mu_rec - mu_real)**2)

def integrate_flow_to_one(model, z0):
    device = z0.device
    def ode_func(t, z):
        t_batch = t.expand(z.size(0))
        return model.flow_model(z, t_batch)
    t_span = torch.tensor([0.0, 1.0], device=device)
    return odeint(ode_func, z0, t_span, method='dopri5')[1]

def pushforward_consistency(model, batch_size, device):
    z0 = torch.randn(batch_size, model.latent_dim, device=device)
    z1 = integrate_flow_to_one(model, z0)          # flow output at t=1
    x_gen = model.decoder(z1)
    mu_gen = _encode_mu(model, x_gen)
    return torch.mean((mu_gen - z1.detach())**2)

def corr_loss(x, xhat, n_genes_sub=256):
    if x.size(1) > n_genes_sub:
        idx = torch.randperm(x.size(1), device=x.device)[:n_genes_sub]
        x, xhat = x[:, idx], xhat[:, idx]
    x     = x - x.mean(0, keepdim=True)
    xhat  = xhat - xhat.mean(0, keepdim=True)
    x     = x / (x.std(0, keepdim=True) + 1e-6)
    xhat  = xhat / (xhat.std(0, keepdim=True) + 1e-6)
    C     = (x.T @ x) / (x.size(0) - 1)
    Chat  = (xhat.T @ xhat) / (xhat.size(0) - 1)
    return torch.mean((C - Chat)**2)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchsde


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, hn = self.gru(inp)
        out = self.lin(out)

        return out

class LatentModel(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentModel, self).__init__()
        self.dt = 0.02
        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        self.f_net = nn.Sequential(
            nn.Linear(latent_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
        self.g_net = nn.Sequential(
            nn.Linear(latent_size+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)
                m.bias.data.fill_(0.01)

    def f(self, t, z):
        t = t.expand(z.size(0), 1).to("cuda")
        tz = torch.cat([t, z], dim=1)
        return self.f_net(tz)

    def g(self, t, z):
        t = t.expand(z.size(0), 1).to("cuda")
        tz = torch.cat([t, z], dim=1)
        return self.g_net(tz)

    def forward(self, xs, ts, ext_ts, method="euler"):
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, ext_ts, dt=self.dt, logqp=False, method=method)

        _xs = self.projector(zs)
        recon_xs = _xs[0::2][1:-1]
        interp_xs = _xs[1::2][1:-1]

        recon_loss = F.mse_loss(recon_xs, xs)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)

        return recon_loss, logqp0, interp_xs

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'f'}, dt=self.dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs
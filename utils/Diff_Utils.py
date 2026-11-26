
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn.functional as F
from tqdm import tqdm
from contextlib import nullcontext

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        # useful inverses
        self.sqrt_recip_alphas_cumprod  = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a, t, x_shape):
        b = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    # q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # x0 = (x_t - sqrt(1-a_bar)*eps) / sqrt(a_bar)
    def predict_start_from_noise(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    # q(x_{t-1} | x_t, x0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        logvar = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, logvar

    # p(x_{t-1}|x_t) using eps-predicting model
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # model predicts eps
        eps_hat = model(x_t, t)

        # reconstruct x0
        x0_hat = self.predict_start_from_noise(x_t, t, eps_hat)

        if clip_denoised:
            x0_hat = x0_hat.clamp(-1., 1.)

        model_mean, posterior_var, posterior_logvar = self.q_posterior_mean_variance(x0_hat, x_t, t)
        return model_mean, posterior_var, posterior_logvar, x0_hat, eps_hat

    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        model_mean, _, model_logvar, x0_hat, _ = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)

        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        x_prev = model_mean + nonzero_mask * (0.5 * model_logvar).exp() * noise
        return x_prev, x0_hat

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device=None, clip_denoised=True, return_x0=False):
        if device is None:
            device = next(model.parameters()).device
        b = shape[0]
        x = torch.randn(shape, device=device)  # x_T

        x0_hat = None
        for i in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps, desc="DDPM sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x, x0_hat = self.p_sample(model, x, t, clip_denoised=clip_denoised)

        return x0_hat if return_x0 else x

    # ----------------------------
    # TRAINING: predict noise eps
    # ----------------------------
    def train_losses(self, model, x_start, t, loss_type="l2"):
        eps = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=eps)

        eps_hat = model(x_t, t)

        if loss_type == "l1":
            loss = F.l1_loss(eps_hat, eps)
        else:
            loss = F.mse_loss(eps_hat, eps)

        return eps_hat, loss

    # ----------------------------
    # DDIM (eps-predicting)
    # ----------------------------
    @torch.no_grad()
    def ddim_sample_from_tstart(
        self, model, shape,
        t_start=200, K=50, eta=0.0,
        device=None, clip_denoised=True,
        x_start=None
    ):
        """
        DDIM sampling for eps-predicting model.

        If x_start is provided (clean x0), initialize x_{t_start} = q_sample(x0,t_start).
        Otherwise start from pure noise.
        Returns: final x0_hat
        """
        if device is None:
            device = next(model.parameters()).device
        B = shape[0]

        ts = torch.linspace(t_start, 0, steps=K, device=device).round().long()
        ts = torch.unique_consecutive(ts)

        # init x_{t_start}
        if x_start is None:
            x = torch.randn(shape, device=device)
        else:
            t0 = torch.full((B,), int(ts[0].item()), device=device, dtype=torch.long)
            eps0 = torch.randn_like(x_start)
            x = self.q_sample(x_start, t0, noise=eps0)

        x0_hat = None

        for k in tqdm(range(len(ts) - 1), desc=f"DDIM t={t_start}->0", total=len(ts) - 1):
            t = ts[k].item()
            t_next = ts[k + 1].item()

            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            tnext_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

            ab_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

            # predict eps and x0
            eps_hat = model(x, t_batch)
            x0_hat = self.predict_start_from_noise(x, t_batch, eps_hat)
            if clip_denoised:
                x0_hat = x0_hat.clamp(-1., 1.)

            # DDIM sigma
            sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12) * (1 - ab_t / (ab_next + 1e-12)))
            noise = torch.randn_like(x) if eta > 0 else 0.0

            # update
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1 - ab_next - sigma**2) * eps_hat + sigma * noise

        return x0_hat

    def ddim_sample_from_tstartLOOP(
            self, model, shape,
            t_start=200, K=50, eta=0.0,
            device=None, clip_denoised=True,
            x_start=None, noise = None
        ):
            """
            DDIM sampling for eps-predicting model.

            If x_start is provided (clean x0), initialize x_{t_start} = q_sample(x0,t_start).
            Otherwise start from pure noise.
            Returns: final x0_hat
            """
            if device is None:
                device = next(model.parameters()).device
            B = shape[0]

            ts = torch.linspace(t_start, 0, steps=K, device=device).round().long()
            ts = torch.unique_consecutive(ts)

            # init x_{t_start}
            if x_start is None:
                x = torch.randn(shape, device=device)
            else:
                t0 = torch.full((B,), int(ts[0].item()), device=device, dtype=torch.long)
                #eps0 = torch.randn_like(x_start)
                eps0 = noise
                x = self.q_sample(x_start, t0, noise=eps0)

            x0_hat = None

            #for k in tqdm(range(len(ts) - 1), desc=f"DDIM t={t_start}->0", total=len(ts) - 1):
            for k in range(len(ts) - 1):

                t = ts[k].item()
                t_next = ts[k + 1].item()

                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                tnext_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

                ab_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
                ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

                # predict eps and x0
                eps_hat = model(x, t_batch)
                x0_hat = self.predict_start_from_noise(x, t_batch, eps_hat)
                if clip_denoised:
                    x0_hat = x0_hat.clamp(-1., 1.)

                # DDIM sigma
                sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12) * (1 - ab_t / (ab_next + 1e-12)))
                noise = torch.randn_like(x) if eta > 0 else 0.0

                # update
                x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1 - ab_next - sigma**2) * eps_hat + sigma * noise

            return x0_hat

    def ddim_x0_from_v0_and_noise_epsmodel(
        self, netD, v0_p, noise,
        t_start=100, K=10, clip_denoised=True):
        """
        Differentiable DDIM for eps-predicting model.
        - v0_p : clean x0 patches (B,1,w1,w2)  [-1,1]
        - noise: optimizable eps (B,1,w1,w2)
        Returns:
          x0_hat (B,1,w1,w2)
        """
        device = v0_p.device
        B = v0_p.shape[0]

        # DDIM schedule: t_start -> 0 in K steps
        ts = torch.linspace(t_start, 0, steps=K, device=device).round().long()
        ts = torch.unique_consecutive(ts)  # remove duplicates

        # Initialize x at t_start using your optimized noise (eps)
        t0 = torch.full((B,), int(ts[0].item()), device=device, dtype=torch.long)
        x = self.q_sample(v0_p, t0, noise=noise)  # x_{t_start}

        x0_hat = None

        # Deterministic DDIM: eta = 0
        for k in range(len(ts) - 1):
            t     = ts[k].item()
            t_next = ts[k + 1].item()

            t_batch     = torch.full((B,), t, device=device, dtype=torch.long)
            tnext_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

            ab_t    = self._extract(self.alphas_cumprod, t_batch, x.shape)
            ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

            # eps prediction
            eps_hat = netD(x, t_batch)

            # x0 reconstruction
            x0_hat = self.predict_start_from_noise(x, t_batch, eps_hat)
            if clip_denoised:
                x0_hat = x0_hat.clamp(-1.0, 1.0)

            # DDIM update (eta=0): x_{t_next}
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1.0 - ab_next) * eps_hat

        return x0_hat
    
    

    def ddim_x0_from_xt_epsmodelFWI(self, model, x_t, t_start=100, K=20, eta=0.0, clip_denoised=True):
        """
        Differentiable DDIM sampling (eps-predicting).
        Starts from provided x_t at timestep t_start and returns x0_hat.
        model: netD_ema (frozen)
        x_t: (B,1,H,W) requires grad if you want to optimize upstream params (noise)
        """
        device = x_t.device
        B = x_t.size(0)

        # timesteps schedule from t_start -> 0
        ts = torch.linspace(t_start, 0, steps=K, device=device).round().long()
        ts = torch.unique_consecutive(ts)

        x = x_t
        x0_hat = None

        for k in range(len(ts) - 1):
            t = ts[k].item()
            t_next = ts[k + 1].item()

            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            tnext_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

            ab_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

            eps_hat = model(x, t_batch)
            x0_hat = self.predict_start_from_noise(x, t_batch, eps_hat)

            if clip_denoised:
                x0_hat = x0_hat.clamp(-1., 1.)

            if eta == 0.0:
                # deterministic DDIM
                x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1 - ab_next) * eps_hat
            else:
                sigma = eta * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12) * (1 - ab_t / (ab_next + 1e-12)))
                noise = torch.randn_like(x)
                x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1 - ab_next - sigma**2) * eps_hat + sigma * noise

        return x0_hat
    
    def xt_from_v0_and_noise(self, v0_p, noise, t_start):
        """
        v0_p: (B,1,H,W) clean initial model in [-1,1]
        noise: (B,1,H,W) optimizable tensor
        returns x_tstart
        """
        B = v0_p.size(0)
        t0 = torch.full((B,), t_start, device=v0_p.device, dtype=torch.long)
        x_t = self.q_sample(v0_p, t0, noise=noise)
        return x_t




    @torch.no_grad()
    def ddim_invert_x0_to_xt(
        self,
        model,
        x0,
        t_start=100,
        K=200,
        clip_denoised=False,   # IMPORTANT for identity test
        use_amp=False,         # IMPORTANT for identity test
        amp_dtype=torch.float16,
    ):
        device = x0.device
        B = x0.shape[0]
        model.eval()

        ts = torch.linspace(0, t_start, steps=K, device=device).round().long()
        ts = torch.unique_consecutive(ts)

        ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()

        x = x0
        with ctx:
            for k in range(len(ts) - 1):
                t      = int(ts[k].item())
                t_next = int(ts[k + 1].item())

                t_batch     = torch.full((B,), t, device=device, dtype=torch.long)
                tnext_batch = torch.full((B,), t_next, device=device, dtype=torch.long)

                ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

                eps_hat = model(x, t_batch)
                x0_hat  = self.predict_start_from_noise(x, t_batch, eps_hat)

                if clip_denoised:
                    x0_hat = x0_hat.clamp(-1.0, 1.0)

                # deterministic forward DDIM step
                x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1.0 - ab_next) * eps_hat

        return x, ts   # RETURN ts!



    def ddim_x0_from_xt_with_ts_chunked(
        self,
        model,
        x_t,
        ts,                    # <-- MUST come from inversion
        eta=0.0,
        clip_denoised=False,   # IMPORTANT for identity test
        patch_chunk=2,
        use_checkpoint=True,
        use_amp=False,
        amp_dtype=torch.float16,
    ):
        device = x_t.device
        B = x_t.shape[0]
        model.eval()

        ts_rev = ts.flip(0)  # t_start -> 0

        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=amp_dtype) if (use_amp and device.type == "cuda")
            else nullcontext()
        )

        def _ddim_one_chunk(x_c):
            bc = x_c.shape[0]
            x = x_c
            x0_hat = None

            for k in range(len(ts_rev) - 1):
                t      = int(ts_rev[k].item())
                t_next = int(ts_rev[k + 1].item())

                t_batch     = torch.full((bc,), t, device=device, dtype=torch.long)
                tnext_batch = torch.full((bc,), t_next, device=device, dtype=torch.long)

                ab_t    = self._extract(self.alphas_cumprod, t_batch, x.shape)
                ab_next = self._extract(self.alphas_cumprod, tnext_batch, x.shape)

                eps_hat = model(x, t_batch)
                x0_hat  = self.predict_start_from_noise(x, t_batch, eps_hat)

                if clip_denoised:
                    x0_hat = x0_hat.clamp(-1.0, 1.0)

                if eta == 0.0:
                    x = torch.sqrt(ab_next) * x0_hat + torch.sqrt(1.0 - ab_next) * eps_hat
                else:
                    sigma = eta * torch.sqrt(
                        (1 - ab_next) / (1 - ab_t + 1e-12) * (1 - ab_t / (ab_next + 1e-12))
                    )
                    eps = torch.randn_like(x)
                    x = (
                        torch.sqrt(ab_next) * x0_hat
                        + torch.sqrt(1.0 - ab_next - sigma**2) * eps_hat
                        + sigma * eps
                    )

            return x0_hat

        outs = []
        for s in range(0, B, patch_chunk):
            e = min(s + patch_chunk, B)
            x_c = x_t[s:e]

            with autocast_ctx:
                if use_checkpoint:
                    x0_c = checkpoint(_ddim_one_chunk, x_c, use_reentrant=False)
                else:
                    x0_c = _ddim_one_chunk(x_c)

            outs.append(x0_c)

        return torch.cat(outs, dim=0)
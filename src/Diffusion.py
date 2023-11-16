import torch
import numpy as np
import math
from tqdm import tqdm


class Diffusion:
    """
    Diffusion class for building a generative model using a markov chain. Both
    the mean and the variance of the noise are learned. Edited from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
    """

    def __init__(self, steps, image_shape, var_method='learned',
                 loss_method='mse', betas_method='linear', beta_start=1e-4,
                 beta_end=0.02,
                 device='cpu'):
        """
        :param steps: number of diffusion steps
        :param image_shape: shape of the image
        :param loss_method: Method for calculating the loss
        :param betas_method: method for calculating the noise schedule
        :param beta_start: Initial noise variance
        :param beta_end: Final noise variance
        :param device: device to run the model on
        :param var_method: method for calculating the noise variance
        """
        self.steps = steps
        self.image_shape = image_shape
        self.betas_method = betas_method
        self.loss_method = loss_method
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.var_method = var_method

        self.betas = self._get_betas()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.insert(self.alphas_cumprod[:-1], 0, 1.0)
        self.alphas_cumprod_next = np.insert(self.alphas_cumprod[:-1], -1, 0.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev)
                                   / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.insert(self.posterior_variance[1:], 0,
                      self.posterior_variance[1]))
        self.posterior_mean_coef1 = (self.betas * np.sqrt(
            self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod))

    def _get_betas(self):
        """
        Calculates the noise schedule.
        """
        if self.betas_method == 'linear':
            betas = np.linspace(self.beta_start, self.beta_end, self.steps)
        elif self.betas_method == 'cosine':
            alpha_f = lambda t: (math.cos(0.5 * math.pi * (t + 0.008) /
                                          1.008) ** 2.)
            betas = []
            for i in range(self.steps):
                t1 = i / self.steps
                t2 = (i + 1) / self.steps
                betas.append(min(1 - alpha_f(t2) / alpha_f(t1), self.beta_end))
            betas = np.array(betas)
        else:
            raise ValueError('Betas method not supported.')
        return betas

    def q_sample(self, x_0, t, eps=None):
        """
        Samples from the model distribution at timestep t, i.e. q(x_t|x_0).
        :param x_0: Initial image
        :param t: Timestep
        :param eps: Noise to sample from. If None, it is sampled from a normal
        """
        if eps is None:
            eps = torch.randn_like(x_0)
        x_t = (_to_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
               _to_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) *
               eps)
        return x_t

    def q_posterior_mean_var(self, x_0, x_t, t):
        """
        Calculates the mean and variance of the diffusion posterior, i.e.
        q(x_{t-1}|x_t, x_0).
        :param x_0: Initial image
        :param x_t: Image at timestep t
        :param t: Timestep
        :return: posterior mean and variance and clipped log variance
        """

        mean = (_to_tensor(self.posterior_mean_coef1, t, x_0.shape) * x_0
                + _to_tensor(self.posterior_mean_coef2, t, x_0.shape) * x_t)
        var = _to_tensor(self.posterior_variance, t, x_0.shape)
        var_log_clipped = _to_tensor(self.posterior_log_variance_clipped, t,
                                     x_0.shape)
        return mean, var, var_log_clipped

    def p_mean_var(self, model, x_t, t, model_kwargs=None, clip_noise=False):
        """
        Calculates the mean and variance of predicted reverse preocess, i.e.
        p(x_{t-1}|x_t).
        :param model: The model which predicted the mean and variance of the
        reverse process
        :param x_t: Image at timestep t
        :param t: Timestep
        :param model_kwargs: Keyword arguments to pass to the model
        :param clip_noise: Whether to clip the noise to the range [-1, 1]
        :return: mean and variance of the reverse process
        """
        if model_kwargs is None:
            model_kwargs = {}

        model_out = model(x_t, t, **model_kwargs)

        if self.var_method == 'learned':
            assert model_out.shape[1] == 2 * x_t.shape[1]
            model_out, model_var_log = torch.split(model_out, x_t.shape[1],
                                                   dim=1)
            min_log = _to_tensor(self.posterior_log_variance_clipped, t,
                                 x_t.shape)
            max_log = torch.log(_to_tensor(self.betas, t, x_t.shape))
            frac = (model_var_log + 1.) / 2.
            model_var_log = frac * max_log + (1. - frac) * min_log
            mode_var = torch.exp(model_var_log)
        elif self.var_method == 'fixed':
            assert model_out.shape[1] == x_t.shape[1]
            mode_var = _to_tensor(self.betas, t, x_t.shape)
        else:
            raise ValueError('Variance method not supported.')

        def process_xstart(x):
            if clip_noise:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t, t, model_out)
        )
        eps = model_out
        model_mean, _, _ = self.q_posterior_mean_var(
            pred_xstart, x_t, t)

        return model_mean, mode_var, pred_xstart, eps

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Predicts the initial image from the noise.
        :param x_t: Image at timestep t
        :param t: timestep
        :param noise: noise
        :return: Initial image
        """

        return (_to_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _to_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
                * eps)

    def p_sample_step(self, model, x_t, t, model_kwargs=None, clip_noise=True):
        """
        Samples from the reverse process, i.e. p(x_{t-1}|x_t).
        :param model: The model which predicted the mean and variance of the
        reverse process
        :param x_t: Image at timestep t
        :param t: Timestep
        :param model_kwargs: Keyword arguments to pass to the model
        :param clip_noise: Whether to clip the noise to the range [-1, 1]
        :return: Sample from the reverse process and the predicted initial image
        """
        model_mean, model_var, pred_start, _ = self.p_mean_var(
            model, x_t, t, model_kwargs, clip_noise)
        eps = torch.where(t.view(-1, 1, 1, 1) != 0, torch.randn_like(
            model_mean), torch.zeros_like(model_mean))
        sample = model_mean + torch.sqrt(model_var) * eps
        return sample, pred_start

    @torch.no_grad()
    def p_sample_full(self, model, n_samples, model_kwargs=None,
                      clip_noise=True):
        """
        Samples from the reverse process, i.e. p(x_{t-1}|x_t), for all timesteps.
        :param model: The model which predicted the mean and variance of the
        reverse process
        :param n_samples: Number of samples to generate
        :param model_kwargs: Keyword arguments to pass to the model
        :param clip_noise: Whether to clip the noise to the range [-1, 1]
        """
        images = torch.randn((n_samples, *self.image_shape), device=self.device)
        with tqdm(total=self.steps, dynamic_ncols=True) as tq:
            for t in reversed(range(self.steps)):
                tq.update(1)
                time = torch.ones(images.shape[0], dtype=torch.long,
                                  device=self.device) * t
                images, _ = self.p_sample_step(model, images, time,
                                               model_kwargs, clip_noise)
        return images

    def training_loss(self, model, x_0, t, model_kwargs=None, kl_weight=1e-3):
        """
        Calculates the training loss for a single timestep.
        :param model: Model which predicts the mean and variance of the
        reverse process
        :param x_0: Image dataset sample
        :param t: Timestep
        :param model_kwargs: Keyword arguments to pass to the model
        :param kl_weight: Weight of the KL divergence term for both loss
        :return: loss
        """

        if model_kwargs is None:
            model_kwargs = {}

        eps = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, eps)

        if self.loss_method == 'kl':
            loss = self._var_low_bound(model, x_0, x_t, t, model_kwargs)
        elif self.loss_method == 'mse':
            x_t = self.q_sample(x_0, t, eps)
            model_mean, model_var, pred_start, noise = self.p_mean_var(
                model, x_t, t, model_kwargs)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(eps, noise)

        elif self.loss_method == 'both':
            model_out = model(x_t, t, **model_kwargs)
            model_out, model_var_log = torch.split(
                model_out, x_t.shape[1], dim=1)
            loss_fn = torch.nn.MSELoss()
            loss_mse = loss_fn(eps, model_out)

            model = lambda x, t: torch.cat((model_out.detach(),
                                            model_var_log), dim=1)
            loss_kl, _ = self._var_low_bound(model, x_0, x_t, t)
            loss = loss_mse + kl_weight * loss_kl
        else:
            raise ValueError('Loss method not supported.')
        return loss

    def _var_low_bound(self, model, x_0, x_t, t, model_kwargs=None):
        """
        Calculates a term in the variational lower bound for training
        :param model: Model which predicts the mean and variance of the
        reverse process
        :param x_0: Image dataset sample
        :param x_t: Image at time t
        :param t: Timestep
        :param model_kwargs: Keyword arguments to pass to the model
        :return: Variational lower bound term
        """
        true_mean, _, log_true_var = self.q_posterior_mean_var(x_0, x_t, t)
        model_mean, model_var, pred_start, _ = self.p_mean_var(
            model, x_t, t, model_kwargs)
        kl = self._kl_divergence(true_mean, torch.exp(log_true_var),
                                 model_mean, model_var)
        return kl.mean(), pred_start

    @staticmethod
    def _kl_divergence(mean1, var1, mean2, var2):
        """
        Calculates the KL divergence between two normal distributions.
        :param mean1: Mean of the first distribution
        :param var1: Variance of the first distribution
        :param mean2: Mean of the second distribution
        :param var2: Variance of the second distribution
        :return: scaler KL divergence
        """
        return 0.5 * (torch.log(var2 / var1) + (var1 + ( mean1 - mean2)**2.)
                      / var2 - 1.)


def _to_tensor(arr, timestep, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timestep: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timestep.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timestep.device)[timestep].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

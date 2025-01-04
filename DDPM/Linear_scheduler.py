import torch 
import torch.nn as nn
class LinearScehduler:
    def __init__(self, num_timestamps, beta_start, beta_end):
        self.num_timestamps = num_timestamps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timestamps) # To linearly increase BETA from start to end, we will have BETA from 0 to T
        self.alphas = 1. - self.betas 
        self.alpha_cumilative_product = torch.cumprod(self.alphas, dim = 0)
        self.alpha_sqroot_cumilative_prod = torch.sqrt(self.alpha_cumilative_product)
        self.one_minus_alpha_squareroot = torch.sqrt( 1. - self.alpha_cumilative_product)


    def add_noise(self, original_image, noise,t ):
        """
        add noise to the image in the forward process
        the images and noise will be of shape BxCxHxW and a 1D tensor for time stamp 't' of size 'B'
        """
        """
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        shape = original_image.shape 
        batch_size = shape[0]

        alpha_sqrt_cum_prod = self.alpha_sqroot_cumilative_prod[t].reshape(batch_size)
        one_minus_alphs_sqrt = self.one_minus_alpha_squareroot[t].reshape(batch_size)

        for _ in range(len(shape)-1):
            """Reshape aplha sqrt and alpha-1 sqrt to Bx1x1x1"""
            alpha_sqrt_cum_prod = alpha_sqrt_cum_prod.unsqueeze(-1)
            one_minus_alphs_sqrt = one_minus_alphs_sqrt.unsqueeze(-1)
        return alpha_sqrt_cum_prod*original_image + one_minus_alphs_sqrt*noise

    def reverse_process(self, xt, noise_predicted, t):
        """
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return: tuple of (mean, image), it returns the predicted mean of the distribution and the predicted denoised image
        """
        x0 = (xt - (self.one_minus_alpha_squareroot[t]*noise_predicted)) / self.alpha_sqroot_cumilative_prod[t]

        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t]*noise_predicted) / self.alpha_sqroot_cumilative_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])

        if t==0:
            return mean, x0
        else:
            variance = (1. - self.alphas[t]) * (1.- self.alpha_cumilative_product[t])
            variance = variance / (1. - self.alphas[t])
            sigma = variance ** 0.5 
            z = torch.randn(xt.shape).to(xt.device)
            #return the sample from the distribution using Reparameterization trick
            return mean + sigma*z, x0

    
        
        

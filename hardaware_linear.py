import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.nn import functional as f
from torch.nn.parameter import Parameter

from utils.settings import settings


class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, LRS=2e3, HRS=10e3, a_prog=-6.24e-4,
                 b_prog=0.691, offset_mean=-0.589, offset_std=0.339, failure_mean_LRS=5.73e-4, failure_std_LRS=9.9e-5,
                 ratio_failure_LRS=0.005, ratio_failure_HRS=0.005, pi=0.5, prior_sigma1=5, prior_sigma2=5,
                 min_conductance=((1 / 1e5) * 1e6),
                 scheme="DoubleColumn", device=None, dtype=None) -> None:
        """
        Initializes the network layer with the input parameters.
        Parameters
        ----------
        in_features: number of inputs to the layer
        out_features: number of outputs to the layer
        bias: Boolean, if True, there is a bias matrix
        LRS: Low Resisitve State in Ohms
        HRS: High Resistive State in Ohms
        a_prog: "a" parameter for the standard deviation of the normal distribution as expressed in the article's figure#2: y = a*G + b
        b_prog: "b" parameter for the standard deviation of the normal distribution as expressed in the article's figure#2: y = a*G + b
        offset_mean: mean of the normal distribution from which the offset is sampled
        offset_std: standard deviation of the distribution from which the offset is sampled
        failure_mean_LRS: mean of the distribution for LRS failures
        failure_std_LRS: standard deviation of the distribution for LRS failures
        ratio_failure_LRS: the ratio for LRS failures
        ratio_failure_HRS: the ratio for HRS failures
        min_conductance: The minimum conductance for sampling from [HRS, MIN_CONDUCTANCE] for HRS failures
        scheme: DoubleColumn or SingleColumn scheme
        device: CPU or CUDA (Not yet implemented)
        dtype: dtype of the ouputs
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.scheme = scheme  # Decide if the memories are to be seperated in positive and negative devices or single column scheme
        self.adj_sigma_use = settings.adj_sigma  # When using an ELBO loss, whether or not to use the sigma of the adjacent memories as a component for the normal distributions to evaluate the kullback leibler divergence. The normal distribution fit on the subdatabases is imperfect, therefore, it might be better to not use this (probably is better to use it however)
        self.sigma = None  # Initialization to nothing
        self.sigma_adj = None  # Initialization to nothing
        self.w_offset = None  # Initialization to nothing
        self.b_offset = None  # Initialization to nothing
        self.a = a_prog  # f(g) = a*g + b variability with respect to conductive level (a component)
        self.b = b_prog  # f(g) = a*g + b variability with respect to conductive level (b component)
        self.max_mask_w = 0
        self.substitution = []
        # dictionary of recorded drifts
        delta_dicts = open("drift_dicts/delta_dicts.pkl", "rb")
        # dictionary of recorded standard deviations on drifts sub-databases
        sigma_delta_dicts = open("drift_dicts/sigma_delta_dicts.pkl", "rb")
        # dictionary of recorded means on drifts sub-databases
        mu_delta_dicts = open("drift_dicts/mu_delta_dicts.pkl", "rb")
        self.delta_dicts = pickle.load(delta_dicts)
        self.delta_sigma_dicts = pickle.load(sigma_delta_dicts)
        self.delta_mu_dicts = pickle.load(mu_delta_dicts)
        delta_dicts.close()
        sigma_delta_dicts.close()  # dictionary of recorded standard deviations
        mu_delta_dicts.close()  # dictionary of recorded means
        self.no_variability = False  # to see behaviour without variability
        self.bbyb = settings.bbyb  # if bayes by backprop is to be used
        self.offset_std = offset_std  # std on the offset sampling
        self.off_mean = offset_mean  # mean on the offset sampling
        self.min_conductance = min_conductance
        self.max_HRS_conductance = 1 / HRS * 1e6  # HRS conductance used for randomly sampling HRS failures
        self.ratio_failure = ratio_failure_LRS + ratio_failure_HRS  # total failure ratio used for random assignment
        self.ratio_failure_LRS = ratio_failure_LRS
        self.ratio_failure_HRS = ratio_failure_HRS
        self.max_cond = (1 / LRS) / 1e-6  # Maximum conductance of the range
        self.min_cond = (1 / HRS) / 1e-6  # Minimum conductance of the range
        self.log_posterior_temp = None
        self.w_scaler = MinMaxScaler(feature_range=(self.min_cond, self.max_cond))  # scaler for weight singlecolumn
        self.w_scaler_dc = MinMaxScaler(feature_range=(
            self.min_cond - self.max_cond, self.max_cond - self.min_cond))  # scaler for weights double column scheme
        self.in_features = in_features
        self.out_features = out_features
        self.training = True  # Boolean informing if the layer is still training or not
        self.mask_w = []
        self.mask_b = []
        self.kld = 0  # Kullback leibler divergence value
        self.adj_s_w_squared = None  # the squared adjacent sigma value for weight
        self.w_sampler = torch.distributions.Normal(torch.zeros((out_features, in_features)),
                                                    1)  # samples a parameter epsilon for reparametrzation trick

        self.w_failure_sampler = torch.distributions.Normal(torch.ones((out_features, in_features)) * failure_mean_LRS,
                                                            failure_std_LRS)  # Samples a random failed device from a distribution (weight)
        self.w_offset_sampler = torch.distributions.Normal(offset_mean * torch.ones((out_features, in_features)),
                                                           offset_std)  # samples a random offset weight
        self.b_offset_sampler = torch.distributions.Normal(offset_mean * torch.ones(out_features),
                                                           offset_std)  # samples a random offset bias
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs))  # Stores the mu of the distributions
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))  # Create a bias parameter for bais
            self.b_sampler = torch.distributions.Normal(torch.zeros(out_features),
                                                        1)  # samples a parameter epsilon for reparametrization trick
            self.adj_s_b_squared = None  # the squared adjacent sigma value for bias
            self.b_scaler_dc = MinMaxScaler(feature_range=(
                self.min_cond - self.max_cond, self.max_cond - self.min_cond))  # scaler for bias double column scheme
            self.b_scaler = MinMaxScaler(feature_range=(self.min_cond, self.max_cond))  # scaler for bias singlecolumn
            self.b_failure_sampler = torch.distributions.Normal(torch.ones(out_features) * failure_mean_LRS,
                                                                failure_std_LRS)  # samples a random failed device from a distribution (bias)
            self.prior_mu_b = torch.zeros(self.bias.shape)
        else:
            self.register_parameter('bias', None)
        self.prior_mu_w = torch.zeros(self.weight.shape)
        self.prior_sigma_w = torch.ones(self.weight.shape) * math.sqrt(
            (pi ** 2 * prior_sigma1 + (1 - pi) ** 2 * prior_sigma2))
        # Prior distributions for bayes by backprop
        self.prior_dist_w_1 = torch.distributions.Normal(self.prior_mu_w, torch.ones(self.weight.shape))
        self.prior_dist_w_2 = torch.distributions.Normal(self.prior_mu_w, torch.ones(self.weight.shape))
        self.prior_dist_b_1 = torch.distributions.Normal(self.prior_mu_b, torch.ones(self.bias.shape))
        self.prior_dist_b_2 = torch.distributions.Normal(self.prior_mu_b, torch.ones(self.bias.shape))

        # Initialization of useful parameters for variability adding
        self.adj_mu_w = torch.FloatTensor(self.weight.shape)
        self.adj_mu_b = torch.FloatTensor(self.bias.shape)
        self.adj_off_w_p = torch.FloatTensor(self.weight.shape)
        self.adj_off_w_n = torch.FloatTensor(self.weight.shape)
        self.adj_off_b_p = torch.FloatTensor(self.bias.shape)
        self.adj_off_b_n = torch.FloatTensor(self.bias.shape)
        self.adj_s_w = torch.FloatTensor(self.weight.shape)
        self.adj_s_b = torch.FloatTensor(self.bias.shape)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.init_adj_sigma(self.adj_sigma_use)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def weight_sampler_dc(self, weight, scaler, w_off_sampler, w_sampler, w_failure_sampler, adj_p, adj_n, adj_sigma):
        """
        Double column scheme to sample new weights and add variability to them

        :param weight: The weight matrix
        :param scaler: The scaler to convert the weigths onto the desired range
        :param w_off_sampler: the sampler for the offset
        :param w_sampler: sampler for the programming precision
        :param w_failure_sampler: sampler for the failed weights value
        :param adj_p: the adjacent offsets for positive weights
        :param adj_n: the adjacent offsets for negative weights
        :param adj_sigma: the sigma values for the adjacent weights
        :return:
        """
        if len(weight.shape) < 2 or weight.shape[0] == 1:
            n_weight = weight.clone().reshape(-1, 1)
            n_weight[0] = weight.abs().max().detach()
            n_weight[n_weight < 0] = 0
            pos_weight = weight.clone()
            neg_weight = weight.clone() * -1
            pos_weight[pos_weight < 0] = 0
            neg_weight[neg_weight < 0] = 0
            scaler.fit(n_weight.detach().reshape(-1, 1))  # Fit MinMaxScaler between 0 and max weight value
            cond_w_pos = scaler.transform(pos_weight.detach().reshape(-1, 1))
            cond_w_neg = scaler.transform(neg_weight.detach().reshape(-1, 1))
            cond_w_pos = cond_w_pos.squeeze()
            cond_w_neg = cond_w_neg.squeeze()
            sigma_resh = True
            delta_cond = cond_w_pos - cond_w_neg
            final_scaler = MinMaxScaler(feature_range=(delta_cond.min(), delta_cond.max()))
            final_scaler.fit(weight.detach().reshape(-1, 1))
        else:
            n_weight = weight.clone()
            n_weight[0][0] = weight.abs().max().detach()
            n_weight[n_weight < 0] = 0
            scaler.fit(n_weight.detach().reshape(-1, 1))  # Fit MinMaxScaler between LRS and HRS
            pos_weight = weight.clone()
            neg_weight = weight.clone() * -1
            pos_weight[pos_weight < 0] = 0
            neg_weight[neg_weight < 0] = 0
            cond_w_pos = scaler.transform(pos_weight.detach().reshape(-1, 1)).reshape(pos_weight.shape)
            cond_w_neg = scaler.transform(neg_weight.detach().reshape(-1, 1)).reshape(neg_weight.shape)
            cond_w_pos = cond_w_pos.squeeze()
            cond_w_neg = cond_w_neg.squeeze()
            delta_cond = (cond_w_pos - cond_w_neg)
            final_scaler = MinMaxScaler(feature_range=(delta_cond.min(), delta_cond.max()))
            final_scaler.fit(weight.detach().reshape(-1, 1))
            sigma_resh = False
        self.w_offset = w_off_sampler.sample()
        # Programming offset, variability + adjacent effect step
        offset_pos = (self.w_offset * cond_w_pos) / 100
        cond_w_pos = np.add(cond_w_pos, offset_pos)

        self.w_offset = w_off_sampler.sample()
        # Programming offset, variability + adjacent effect step
        offset_neg = (self.w_offset * cond_w_neg) / 100
        cond_w_neg = np.add(cond_w_neg, offset_neg)
        self.w_offset = w_off_sampler.loc * (cond_w_pos / 100 - cond_w_neg / 100) * ((
                                                                                             weight.max() - weight.min()) / (
                                                                                             self.max_cond - self.min_cond)).detach()

        sigma_p = self.a * cond_w_pos + self.b
        sigma_n = self.a * cond_w_neg + self.b

        sigma_p = sigma_p / 100 * cond_w_pos  # get the sigma in conductance value
        sigma_n = sigma_n / 100 * cond_w_neg
        offset_sigma_p = self.offset_std / 100 * cond_w_pos  # get the offset sigma in conductance value
        offset_sigma_n = self.offset_std / 100 * cond_w_neg

        cond_w_pos = np.add(cond_w_pos, sigma_p * w_sampler.sample())  # reparametrization trick
        cond_w_pos = np.add(cond_w_pos, adj_p)

        cond_w_neg = np.add(cond_w_neg, sigma_n * w_sampler.sample())  # reparametrization trick
        cond_w_neg = np.add(cond_w_neg, adj_n)

        # ======== Substitution step by failed devices ======== #
        if self.ratio_failure_LRS != 0 or self.ratio_failure_HRS != 0:  # avoid division by 0
            hrs_mask_pos = torch.FloatTensor(weight.shape).uniform_() <= self.ratio_failure_HRS / (
                    self.ratio_failure_LRS + self.ratio_failure_HRS)
            lrs_mask_pos = torch.bitwise_not(hrs_mask_pos)
            mask_w_pos = torch.FloatTensor(
                weight.shape).uniform_() <= self.ratio_failure  # Create a random weight mask based on the probability of failure of the devices
            hrs_mask_pos = torch.bitwise_and(hrs_mask_pos, mask_w_pos)
            lrs_mask_pos = torch.bitwise_and(lrs_mask_pos, mask_w_pos)

            hrs_mask_neg = torch.FloatTensor(weight.shape).uniform_() <= self.ratio_failure_HRS / (
                    self.ratio_failure_LRS + self.ratio_failure_HRS)
            lrs_mask_neg = torch.bitwise_not(hrs_mask_neg)
            mask_w_neg = torch.FloatTensor(
                weight.shape).uniform_() <= self.ratio_failure  # Create a random weight mask based on the probability of failure of the devices
            hrs_mask_neg = torch.bitwise_and(hrs_mask_neg, mask_w_neg)
            lrs_mask_neg = torch.bitwise_and(lrs_mask_neg, mask_w_neg)

            self.mask_w = torch.logical_or(torch.logical_or(hrs_mask_neg, lrs_mask_neg),
                                           torch.logical_or(hrs_mask_pos, lrs_mask_pos))
            mask_w = self.mask_w
            # Compile some statistics for how much substitution is seen on average
            if len(mask_w) != 1:
                if mask_w.count_nonzero().numpy() * 100 / np.prod(mask_w.shape) > self.max_mask_w:
                    self.max_mask_w = mask_w.count_nonzero().numpy() * 100 / np.prod(mask_w.shape)
                self.substitution.append(mask_w.count_nonzero().numpy() * 100 / len(mask_w))
            cond_w_pos = cond_w_pos.float()
            cond_w_pos[lrs_mask_pos] = w_failure_sampler.sample()[lrs_mask_pos] * 1e6
            cond_w_pos[hrs_mask_pos] = ((self.max_HRS_conductance - self.min_conductance) * torch.rand(weight.shape)[
                hrs_mask_pos] + self.min_conductance)

            cond_w_neg = cond_w_neg.float()
            cond_w_neg[lrs_mask_neg] = w_failure_sampler.sample()[lrs_mask_neg] * 1e6
            cond_w_neg[hrs_mask_neg] = ((self.max_HRS_conductance - self.min_conductance) * torch.rand(weight.shape)[
                hrs_mask_neg] + self.min_conductance)
        if sigma_resh:  # to reshape the sigma
            w_global = cond_w_pos - cond_w_neg
            if settings.elbo:
                self.sigma = (((np.sqrt(
                    sigma_p ** 2 + sigma_n ** 2 + offset_sigma_p ** 2 + offset_sigma_n ** 2 + 2 * adj_sigma)) * (
                                       weight.max() - weight.min())) / (
                                      self.max_cond - self.min_cond)).detach()  # + weight.min() #- (self.min_cond - self.max_cond))
            w_equ = (torch.FloatTensor(final_scaler.inverse_transform(w_global.reshape(-1, 1))))
            delta = w_equ.squeeze() - weight.detach()  # necessary for reparametrization trick
            weight = weight + delta
            # weight[weight >= 0] = weight[weight >= 0] + delta_p[weight >= 0]
            if (len(w_equ.squeeze().shape) != 0 and self.ratio_failure_LRS != 0) and settings.elbo:
                self.sigma[torch.bitwise_and(lrs_mask_neg, lrs_mask_pos)] = 99.44 * (weight.max() - weight.min()) / (
                        self.max_cond - self.min_cond)
                self.sigma[torch.bitwise_and(hrs_mask_neg, hrs_mask_pos)] = 14 * (weight.max() - weight.min()) / (
                        self.max_cond - self.min_cond)
        else:
            w_global = cond_w_pos - cond_w_neg
            if settings.elbo:
                self.sigma = (((np.sqrt(
                    sigma_p ** 2 + sigma_n ** 2 + offset_sigma_p ** 2 + offset_sigma_n ** 2 + 2 * adj_sigma)) * (
                                       weight.max() - weight.min())) / (
                                      self.max_cond - self.min_cond)).detach()  # + weight.min()
            w_equ = torch.FloatTensor(final_scaler.inverse_transform(w_global.reshape(-1, 1))).reshape(weight.shape)
            delta = w_equ - weight.detach()
            weight = weight + delta
            if (self.ratio_failure_LRS != 0 or self.ratio_failure_HRS != 0) and settings.elbo:
                self.sigma[torch.bitwise_and(lrs_mask_neg, lrs_mask_pos)] = 99.44 * (weight.max() - weight.min()) / (
                        self.max_cond - self.min_cond)
                self.sigma[torch.bitwise_and(hrs_mask_neg, hrs_mask_pos)] = 14 * (weight.max() - weight.min()) / (
                        self.max_cond - self.min_cond)
        return weight

    def weight_sampler_sc(self, weight, scaler, w_off_sampler, w_sampler, w_failure_sampler, adj, adj_sigma):
        """
        Double column scheme to sample new weights and add variability to them

        :param weight: The weight matrix
        :param scaler: The scaler to convert the weigths onto the desired range
        :param w_off_sampler: the sampler for the offset
        :param w_sampler: sampler for the programming precision
        :param w_failure_sampler: sampler for the failed weights value
        :param adj: the adjacent offsets
        :param adj_sigma: the sigma values for the adjacent weights
        :return:
        """
        self.mu = weight
        if len(weight.shape) < 2:
            # Fit MinMaxScaler between LRS and HRS
            scaler.fit(weight.detach().reshape(-1, 1))
            cond_w = scaler.transform(weight.detach().reshape(-1, 1))
            cond_w = cond_w.squeeze()
            sigma_resh = True
        else:
            scaler.fit(weight.detach().reshape(-1, 1))  # Fit MinMaxScaler between LRS and HRS
            cond_w = scaler.transform(weight.detach().reshape(-1, 1))  # transform the weights
            sigma_resh = False
        self.w_offset = w_off_sampler.sample()
        # Programming offset, variability + adjacent effect step
        cond_w = np.subtract(cond_w, (self.w_offset * cond_w) / 100)
        self.w_offset = (w_off_sampler.loc * cond_w) / 100 * ((
                                                                      weight.max() - weight.min()) / (
                                                                      self.max_cond - self.min_cond)).detach()
        sigma = self.a * cond_w + self.b
        sigma = sigma / 100 * cond_w  # get the sigma in conductance value
        offset_sigma = self.offset_std / 100 * cond_w  # get the offset sigma in conductance value
        cond_w = np.add(cond_w, sigma * w_sampler.sample())  # reparametrization trick
        cond_w = np.add(cond_w, adj)
        # Substitution step
        if self.ratio_failure_LRS != 0 or self.ratio_failure_HRS != 0:  # avoid division by 0
            hrs_mask = torch.FloatTensor(weight.shape).uniform_() <= self.ratio_failure_HRS / (
                    self.ratio_failure_LRS + self.ratio_failure_HRS)
            lrs_mask = torch.bitwise_not(hrs_mask)
            mask_w = torch.FloatTensor(
                weight.shape).uniform_() <= self.ratio_failure  # Create a random weight mask based on the probability of failure of the devices
            hrs_mask = torch.bitwise_and(hrs_mask, mask_w)
            lrs_mask = torch.bitwise_and(lrs_mask, mask_w)
            self.mask_w = mask_w
            cond_w = cond_w.float()
            try:
                cond_w[lrs_mask] = w_failure_sampler.sample()[lrs_mask] * 1e6
                cond_w[hrs_mask] = ((self.max_HRS_conductance - self.min_conductance) * torch.rand(weight.shape)[
                    hrs_mask] + self.min_conductance)
            except:
                print(lrs_mask)
        if sigma_resh:  # to reshape the sigma
            new_ratio = weight.max() - weight.min()
            if new_ratio == 0:
                new_ratio = 1
            self.sigma = (((np.sqrt(
                sigma ** 2 + offset_sigma ** 2 + adj_sigma)) * (
                               new_ratio)) / (self.max_cond - self.min_cond)).detach()
            w_equ = (torch.FloatTensor(scaler.inverse_transform(cond_w.reshape(-1, 1))))
            delta = w_equ.squeeze() - weight.detach()  # necessary for reparametrization trick
            weight = weight + delta
            if (len(w_equ.squeeze().shape) != 0 and self.ratio_failure_LRS != 0) and settings.elbo:
                self.sigma[lrs_mask] = 99.44 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
                self.sigma[hrs_mask] = 14 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
        else:
            new_ratio = weight.max() - weight.min()
            if new_ratio == 0:
                new_ratio = 1
            self.sigma = (((np.sqrt(
                sigma ** 2 + offset_sigma ** 2 + adj_sigma)) * (
                               new_ratio)) / (self.max_cond - self.min_cond)).detach()
            w_equ = (torch.FloatTensor(scaler.inverse_transform(cond_w)))
            delta = w_equ - weight.detach()
            weight = weight + delta
            if (self.ratio_failure_LRS != 0 or self.ratio_failure_HRS != 0) and settings.elbo:
                try:
                    self.sigma[lrs_mask] = 99.44 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
                    self.sigma[hrs_mask] = 14 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
                    self.mu[hrs_mask] = 45 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
                    self.mu[lrs_mask] = 572.99 * (weight.max() - weight.min()) / (self.max_cond - self.min_cond)
                except:
                    print("error")
        return weight

    def init_adj_sigma(self, adj_sigma_use):
        """
        Initializes the adjacent effect detuning sigmas if elbo is to be used. However, note that this is
        Parameters
        ----------
        adj_sigma_use

        Returns
        -------

        """
        x = 0
        y = 0
        for i in range(torch.numel(self.weight), 0, -1):
            if len(self.weight.shape) > 1 and self.weight.shape[0] > 1:
                if i != 0:
                    self.adj_s_w[y][x] = self.delta_sigma_dicts[str(i)] * 1e6
                else:
                    self.adj_s_w[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                self.adj_s_w[0][x] = self.delta_sigma_dicts[str(i)] * 1e6
                x += 1
        self.adj_s_w[-1][-1] = 0
        x = 0
        y = 0
        for i in range(torch.numel(self.bias), 0, -1):
            if len(self.bias.shape) > 1 and self.bias.shape[0] > 1:
                if i != 0:
                    self.adj_s_b[y][x] = self.delta_sigma_dicts[str(i)] * 1e6
                else:
                    self.adj_s_b[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                if len(self.adj_s_b.shape) > 1:
                    self.adj_s_b[0][x] = self.delta_sigma_dicts[str(i)] * 1e6
                else:
                    self.adj_s_b[x] = self.delta_sigma_dicts[str(i)] * 1e6
                x += 1
        if len(self.adj_s_b.shape) > 1:
            self.adj_s_b[-1][-1] = 0
        else:
            self.adj_s_b[-1] = 0
        if adj_sigma_use:
            self.adj_s_b_squared = self.adj_s_b ** 2
            self.adj_s_w_squared = self.adj_s_w ** 2
        else:
            self.adj_s_b_squared = torch.zeros(self.adj_s_b.shape)
            self.adj_s_w_squared = torch.zeros(self.adj_s_w.shape)

    def init_adj_mu(self):
        """
        Initiates the mu of adjacent devices
        Returns
        -------
        """
        x = 0
        y = 0
        for i in range(torch.numel(self.weight), 0, -1):
            if len(self.weight.shape) > 1 and self.weight.shape[0] > 1:
                if i != 0:
                    self.adj_mu_w[y][x] = self.delta_mu_dicts[str(i)] * 1e6
                else:
                    self.adj_mu_w[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                self.adj_mu_w[0][x] = self.delta_mu_dicts[str(i)] * 1e6
                x += 1
        self.adj_mu_w[-1][-1] = 0
        x = 0
        y = 0
        for i in range(torch.numel(self.bias), 0, -1):
            if len(self.bias.shape) > 1 and self.bias.shape[0] > 1:
                if i != 0:
                    self.adj_mu_b[y][x] = self.delta_mu_dicts[str(i)] * 1e6
                else:
                    self.adj_mu_b[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                if len(self.adj_mu_b.shape) > 1:
                    self.adj_mu_b[0][x] = self.delta_mu_dicts[str(i)] * 1e6
                else:
                    self.adj_mu_b[x] = self.delta_mu_dicts[str(i)] * 1e6
                x += 1
        if len(self.adj_mu_b.shape) > 1:
            self.adj_mu_b[-1][-1] = 0
        else:
            self.adj_mu_b[-1] = 0

    def sample_adj_effect_offsets(self):
        """
        Code to sample new random detuning effects to neighbours programming from the database.
        """
        x = 0
        y = 0
        for i in range(torch.numel(self.weight), 0, -1):
            if len(self.weight.shape) > 1 and self.weight.shape[0] > 1:
                if i != 0:
                    self.adj_off_w_p[y][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                    self.adj_off_w_n[y][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                    if abs(self.adj_off_w_p[y][x]) > 4 * self.delta_sigma_dicts[
                        str(i)] * 1e6:  # since probability of getting more than 4 * std is less thank 0.0001
                        self.adj_off_w_p[y][x] = 60
                    if abs(self.adj_off_w_n[y][x]) > 4 * self.delta_sigma_dicts[str(i)] * 1e6:
                        self.adj_off_w_n[y][x] = 60
                else:
                    self.adj_off_w_p[y][x] = 0
                    self.adj_off_w_n[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                self.adj_off_w_p[0][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                self.adj_off_w_n[0][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                # since probability of getting more than 4 * std is less thank 0.0001
                if abs(self.adj_off_w_p[y][x]) > 4 * self.delta_sigma_dicts[str(i)] * 1e6:
                    self.adj_off_w_p[0][x] = 60
                if abs(self.adj_off_w_n[y][x]) > 4 * self.delta_sigma_dicts[str(i)] * 1e6:
                    self.adj_off_w_n[0][x] = 60
                x += 1
        self.adj_off_w_p[-1][-1] = 0
        self.adj_off_w_n[-1][-1] = 0
        x = 0
        y = 0
        for i in range(torch.numel(self.bias), 0, -1):
            if len(self.bias.shape) > 1 and self.bias.shape[0] > 1:
                if i != 0:
                    self.adj_off_b_p[y][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                    self.adj_off_b_n[y][x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                    if abs(self.adj_off_b_p[y][x]) > 4 * self.delta_sigma_dicts[
                        str(i)] * 1e6:  # since probability of getting more than 4 * std is less thank 0.0001
                        self.adj_off_b_p[0][x] = 60
                    if abs(self.adj_off_b_n[y][x]) > 4 * self.delta_sigma_dicts[str(i)] * 1e6:
                        self.adj_off_b_n[0][x] = 60
                else:
                    self.adj_off_b_p[y][x] = 0
                    self.adj_off_b_n[y][x] = 0
                y += 1
                if y == 8:
                    y = 0
                    x += 1
            else:
                self.adj_off_b_p[x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                self.adj_off_b_n[x] = random.choice(self.delta_dicts[str(i)]) * 1e6
                x += 1
        self.adj_off_b_p[-1] = 0
        self.adj_off_b_n[-1] = 0

    def forward(self, input: Tensor) -> Tensor:
        """
        forward pass of the Linear layer
        :param input: input to perform forward passes on
        :return:
        """
        if self.no_variability:  # bypass all the variability adding
            new_weight = self.weight
            new_bias = self.bias
        else:
            self.kld = 0  # RESET kullback leibler divergence
            self.sample_adj_effect_offsets()  # sample random offsets due to neighbour programming before hand
            # ====== weight forward =======
            new_weight, self.mask_w_weight, sc_w = self.w_forward(self.weight, self.w_scaler, self.w_scaler_dc,
                                                                  self.w_offset_sampler, self.w_sampler,
                                                                  self.w_failure_sampler, self.adj_off_w_p,
                                                                  self.adj_off_w_n,
                                                                  self.adj_s_w_squared, self.prior_dist_w_1,
                                                                  self.prior_dist_w_2)
            # ====== bias forward =======
            new_bias, self.mask_w_bias, sc_b = self.w_forward(self.bias, self.b_scaler, self.b_scaler_dc,
                                                              self.b_offset_sampler, self.b_sampler,
                                                              self.b_failure_sampler, self.adj_off_b_p,
                                                              self.adj_off_b_n,
                                                              self.adj_s_b_squared, self.prior_dist_b_1,
                                                              self.prior_dist_b_2)
        return f.linear(input, new_weight, new_bias)  # perform Linear layer operation

    def w_forward(self, weight, w_scaler, w_scaler_dc, w_offset_sampler, w_sampler, LRS_failure_sampler, adj_off_p,
                  adj_off_n, adj_s_squared, prior_dist_1, prior_dist_2):
        """
        Forward call sub-function including sampling of weights
        weight: the weight of biases matrix
        w_scaler: scaler transfering from the weights range to the conductance range
        w_scaler_dc: scaler transfering from the weights range to the delta conductance range (max conductance - min conductance)
        w_offset_sampler: samples a random mean offset to add to each weight
        w_sampler: samples a parameter epsilon from which the sigma reparametrization trick can be done
        LRS_failure_sampler: samples a failed memory in LRS state
        adj_off_p: a matrix of randomly generated offsets for positive memories
        adj_off_n: a matrix of randomly generated offsets for negative memories
        adj_s_squared: a matrix of squared sigmas to be used for distribution establishing
        prior_dist_1: the first element of the slab prior distribution on the weights as presented in "Weight uncertainity in Neural Networks"
        prior_dist_2: the second element of the slab prior distribution on the weights as presented in "Weight uncertainity in Neural Networks"
        :return:
        """
        if ((self.scheme == "SingleColumn" or torch.all(weight >= 0) or torch.all(
                weight <= 0)) and weight.shape != torch.Size(
            [1])):  # Use single column scheme is all weights are positive or if the scheme is deemed as such
            new_weight = self.weight_sampler_sc(weight, w_scaler, w_offset_sampler, w_sampler,
                                                LRS_failure_sampler, adj_off_p, adj_s_squared)
            self.sigma = self.sigma.squeeze()
            sc = True
        elif weight.shape != torch.Size([1]):
            new_weight = self.weight_sampler_dc(weight, w_scaler_dc, w_offset_sampler,
                                                w_sampler,
                                                LRS_failure_sampler, adj_off_p, adj_off_n,
                                                adj_s_squared)
            sc = False
        else:
            sc = True
            self.mask_w = False
            new_weight = weight
            self.sigma = torch.Tensor(0)
            self.w_offset = 0
        mask_w = self.mask_w  # debug purposes
        if settings.elbo and weight.shape != torch.Size([1]):
            if self.bbyb:  # Bayes by backprop procedure
                self.kld += (self.log_posterior(new_weight, weight + self.w_offset, self.sigma,
                                                self.mask_w) - self.log_prior(
                    new_weight,
                    prior_dist_1,
                    prior_dist_2))
            else:  # VAE procedure
                logvar = torch.log(self.sigma ** 2)
                inner_equ = 1 + logvar - new_weight.pow(2) - logvar.exp()
                if inner_equ.ndim > 1:
                    self.kld += torch.mean(-0.5 * torch.sum(inner_equ, dim=1), dim=0)
                else:
                    self.kld += torch.mean(-0.5 * torch.sum(inner_equ, dim=0), dim=0)
        return new_weight, mask_w, sc

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)

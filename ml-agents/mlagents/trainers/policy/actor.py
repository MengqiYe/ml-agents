# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Mengqi.Ye on 10/31/2020
"""

import torch

from pt_mlagents.trainers.distributions import GaussianDistribution


class ContinuousControlActor(torch.nn.Module):
    def __init__(
            self,
            encoded: torch.nn.Module,
            tanh_squash: bool = False,
            reparameterize: bool = False,
            condition_sigma_on_obs: bool = True,
    ):
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :param tanh_squash: Whether to use a tanh function, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy.
        """
        super(ContinuousControlActor, self).__init__()
        self.encoded = encoded
        self.tanh_squash = tanh_squash
        self.reparameterize = reparameterize
        self.condition_sigma_on_obs = condition_sigma_on_obs

        self.distribution = GaussianDistribution(
            self.act_size,
            reparameterize=reparameterize,
            tanh_squash=tanh_squash,
            condition_sigma=condition_sigma_on_obs,
        )

        self.output = None

    def forward(self, x):
        hidden_policy = self.encoded(x)
        self.distribution.forward(hidden_policy)
        self.output_pre = self.distribution.sample
        if self.tanh_squash:
            self.output = self.output_pre.detach()
        else:
            output_post = torch.clamp(-3, 3, self.output_pre) / 3
            self.output = output_post.detach()


        pass

    @property
    def action(self):
        return self.output

    @property
    def action_probs(self):
        return self.distribution.log_probs

    @property
    def entropy(self):
        return self.distribution.entropy


class DiscreteControlActor(torch.nn.Module):
    def __init__(
            self,
            encoded: torch.nn.Module,
            tanh_squash: bool = False,
            reparameterize: bool = False,
            condition_sigma_on_obs: bool = True,
    ):
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        """
        super(DiscreteControlActor, self).__init__()
        self.use_recurrent = True



    def forward(self, encoded):
        if self.use_recurrent:
            
            prev_action_oh = torch.cat([self.prev_action for i in range(len(self.act_size))])
        hidden_policy = torch.cat([encoded, prev_action_oh])

    @property
    def prev_action(self):
        pass

    @property
    def recurrent_in(self):
        pass

    @property
    def recurrent_out(self):
        pass

    @property
    def action_masks(self):
        pass

    @property
    def action(self):
        return self.output

    @property
    def action_probs(self):
        return self.all_log_probs
